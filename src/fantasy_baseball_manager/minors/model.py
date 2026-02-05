"""Gradient boosting models for MLE (Minor League Equivalency) prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


class _HyperparametersDict(TypedDict):
    """Serialized hyperparameters structure."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_samples: int
    subsample: float
    random_state: int


class _MLEStatModelParams(TypedDict):
    """Serialized MLEStatModel structure."""

    stat_name: str
    hyperparameters: _HyperparametersDict
    feature_names: list[str]
    model: LGBMRegressor | None
    is_fitted: bool


class _MLEModelSetParams(TypedDict):
    """Serialized MLEGradientBoostingModel structure."""

    player_type: str
    models: dict[str, _MLEStatModelParams]
    feature_names: list[str]
    training_years: list[int]


@dataclass(frozen=True)
class MLEHyperparameters:
    """Hyperparameters for MLE gradient boosting model.

    These defaults are tuned for the smaller MLE training dataset (~400-600 samples)
    with stronger regularization to prevent overfitting.
    """

    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    min_child_samples: int = 20
    subsample: float = 0.8
    random_state: int = 42


@dataclass
class MLEStatModel:
    """Wrapper around LightGBM for predicting a single MLE stat.

    The model predicts: MLB_rate = f(MiLB_features)
    where features include rate stats, age, level, and sample size.
    """

    stat_name: str
    hyperparameters: MLEHyperparameters = field(default_factory=MLEHyperparameters)
    _model: LGBMRegressor | None = field(default=None, repr=False)
    _feature_names: list[str] = field(default_factory=list, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        """Train the model on feature matrix X and target rates y.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target MLB rates of shape (n_samples,)
            feature_names: Names of features for importance tracking
            sample_weight: Optional sample weights (MLB PA for weighting)
            eval_set: Optional (X_val, y_val) tuple for early stopping evaluation
            early_stopping_rounds: Stop training if no improvement for this many rounds
        """
        import pandas as pd
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation

        hp = self.hyperparameters
        model = LGBMRegressor(
            n_estimators=hp.n_estimators,
            max_depth=hp.max_depth,
            learning_rate=hp.learning_rate,
            min_child_samples=hp.min_child_samples,
            subsample=hp.subsample,
            random_state=hp.random_state,
            verbosity=-1,
        )

        X_df = pd.DataFrame(X, columns=feature_names)

        fit_kwargs: dict = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if eval_set is not None and early_stopping_rounds is not None:
            X_val, y_val = eval_set
            X_val_df = pd.DataFrame(X_val, columns=feature_names)
            fit_kwargs["eval_set"] = [(X_val_df, y_val)]
            fit_kwargs["callbacks"] = [
                early_stopping(stopping_rounds=early_stopping_rounds),
                log_evaluation(period=0),  # Suppress logging
            ]

        model.fit(X_df, y, **fit_kwargs)
        self._model = model
        self._feature_names = list(feature_names)
        self._is_fitted = True
        logger.debug(
            "Trained MLEStatModel for %s on %d samples",
            self.stat_name,
            len(y),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict MLB rates for feature matrix X.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted MLB rates of shape (n_samples,)

        Raises:
            ValueError: If model has not been fitted
        """
        import pandas as pd

        if not self._is_fitted or self._model is None:
            raise ValueError(f"MLEStatModel for {self.stat_name} has not been fitted")
        X_df = pd.DataFrame(X, columns=self._feature_names)
        return self._model.predict(X_df)

    def feature_importances(self) -> dict[str, float]:
        """Return feature importances as a dict mapping feature name to importance.

        Raises:
            ValueError: If model has not been fitted
        """
        if not self._is_fitted or self._model is None:
            raise ValueError(f"MLEStatModel for {self.stat_name} has not been fitted")
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances, strict=True))

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_fitted

    def get_params(self) -> _MLEStatModelParams:
        """Return model state for serialization."""
        return _MLEStatModelParams(
            stat_name=self.stat_name,
            hyperparameters=_HyperparametersDict(
                n_estimators=self.hyperparameters.n_estimators,
                max_depth=self.hyperparameters.max_depth,
                learning_rate=self.hyperparameters.learning_rate,
                min_child_samples=self.hyperparameters.min_child_samples,
                subsample=self.hyperparameters.subsample,
                random_state=self.hyperparameters.random_state,
            ),
            feature_names=self._feature_names,
            model=self._model,
            is_fitted=self._is_fitted,
        )

    @classmethod
    def from_params(cls, params: _MLEStatModelParams) -> MLEStatModel:
        """Reconstruct model from serialized state."""
        hp = MLEHyperparameters(**params["hyperparameters"])
        model = cls(stat_name=params["stat_name"], hyperparameters=hp)
        model._model = params["model"]
        model._feature_names = params["feature_names"]
        model._is_fitted = params["is_fitted"]
        return model


@dataclass
class MLEGradientBoostingModel:
    """Collection of MLEStatModels for all target stats.

    This model predicts MLB-equivalent rates from MiLB features.
    It contains one model per target stat (hr, so, bb, singles, doubles, triples, sb).
    """

    player_type: str = "batter"
    models: dict[str, MLEStatModel] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    training_years: tuple[int, ...] = field(default_factory=tuple)

    def add_model(self, model: MLEStatModel) -> None:
        """Add a trained model for a stat."""
        self.models[model.stat_name] = model

    def predict_rates(self, X: np.ndarray) -> dict[str, float]:
        """Predict MLB rates for all stats from a single feature vector.

        Args:
            X: Feature vector of shape (n_features,) or (1, n_features)

        Returns:
            Dict mapping stat name to predicted MLB rate
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        rates: dict[str, float] = {}
        for stat_name, model in self.models.items():
            if model.is_fitted:
                pred = model.predict(X)
                rates[stat_name] = float(pred[0])

        return rates

    def predict_rates_batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict MLB rates for all stats from multiple feature vectors.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Dict mapping stat name to array of predicted MLB rates
        """
        import numpy as np

        rates: dict[str, np.ndarray] = {}
        for stat_name, model in self.models.items():
            if model.is_fitted:
                rates[stat_name] = model.predict(X)
            else:
                rates[stat_name] = np.zeros(len(X))

        return rates

    def get_stats(self) -> list[str]:
        """Return list of stat names with trained models."""
        return [name for name, model in self.models.items() if model.is_fitted]

    def feature_importances(self, stat_name: str) -> dict[str, float]:
        """Return feature importances for a specific stat.

        Args:
            stat_name: Name of the stat to get importances for

        Returns:
            Dict mapping feature name to importance value

        Raises:
            KeyError: If stat_name not in models
            ValueError: If model for stat_name is not fitted
        """
        if stat_name not in self.models:
            raise KeyError(f"No model for stat: {stat_name}")
        return self.models[stat_name].feature_importances()

    def get_params(self) -> _MLEModelSetParams:
        """Return model set state for serialization."""
        return _MLEModelSetParams(
            player_type=self.player_type,
            models={name: model.get_params() for name, model in self.models.items()},
            feature_names=self.feature_names,
            training_years=list(self.training_years),
        )

    @classmethod
    def from_params(cls, params: _MLEModelSetParams) -> MLEGradientBoostingModel:
        """Reconstruct model set from serialized state."""
        model_set = cls(
            player_type=params["player_type"],
            feature_names=params["feature_names"],
            training_years=tuple(params["training_years"]),
        )
        for name, model_params in params["models"].items():
            model_set.models[name] = MLEStatModel.from_params(model_params)
        return model_set
