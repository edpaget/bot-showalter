"""Gradient boosting models for predicting Marcel projection residuals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelHyperparameters:
    """Hyperparameters for gradient boosting model."""

    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    min_child_samples: int = 20
    subsample: float = 0.8
    random_state: int = 42


@dataclass
class StatResidualModel:
    """Wrapper around LightGBM for predicting residuals of a single stat.

    The model predicts: actual_stat - marcel_projected_stat
    """

    stat_name: str
    hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    _model: LGBMRegressor | None = field(default=None, repr=False)
    _feature_names: list[str] = field(default_factory=list, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:
        """Train the model on feature matrix X and residual targets y.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target residuals of shape (n_samples,)
            feature_names: Names of features for importance tracking
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

        fit_kwargs: dict[str, Any] = {}
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
            "Trained StatResidualModel for %s on %d samples",
            self.stat_name,
            len(y),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict residuals for feature matrix X.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted residuals of shape (n_samples,)

        Raises:
            ValueError: If model has not been fitted
        """
        import pandas as pd

        if not self._is_fitted or self._model is None:
            raise ValueError(f"StatResidualModel for {self.stat_name} has not been fitted")
        X_df = pd.DataFrame(X, columns=self._feature_names)
        return self._model.predict(X_df)

    def feature_importances(self) -> dict[str, float]:
        """Return feature importances as a dict mapping feature name to importance.

        Raises:
            ValueError: If model has not been fitted
        """
        if not self._is_fitted or self._model is None:
            raise ValueError(f"StatResidualModel for {self.stat_name} has not been fitted")
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances, strict=True))

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been trained."""
        return self._is_fitted

    def get_params(self) -> dict[str, Any]:
        """Return model state for serialization."""
        return {
            "stat_name": self.stat_name,
            "hyperparameters": {
                "n_estimators": self.hyperparameters.n_estimators,
                "max_depth": self.hyperparameters.max_depth,
                "learning_rate": self.hyperparameters.learning_rate,
                "min_child_samples": self.hyperparameters.min_child_samples,
                "subsample": self.hyperparameters.subsample,
                "random_state": self.hyperparameters.random_state,
            },
            "feature_names": self._feature_names,
            "model": self._model,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> StatResidualModel:
        """Reconstruct model from serialized state."""
        hp = ModelHyperparameters(**params["hyperparameters"])
        model = cls(stat_name=params["stat_name"], hyperparameters=hp)
        model._model = params["model"]
        model._feature_names = params["feature_names"]
        model._is_fitted = params["is_fitted"]
        return model


@dataclass
class ResidualModelSet:
    """Collection of StatResidualModels for all stats of a player type (batter or pitcher)."""

    player_type: str  # "batter" or "pitcher"
    models: dict[str, StatResidualModel] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    training_years: tuple[int, ...] = field(default_factory=tuple)

    def add_model(self, model: StatResidualModel) -> None:
        """Add a trained model for a stat."""
        self.models[model.stat_name] = model

    def predict_residuals(self, X: np.ndarray) -> dict[str, float]:
        """Predict residuals for all stats from a single feature vector.

        Args:
            X: Feature vector of shape (n_features,) or (1, n_features)

        Returns:
            Dict mapping stat name to predicted residual
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        residuals: dict[str, float] = {}
        for stat_name, model in self.models.items():
            if model.is_fitted:
                pred = model.predict(X)
                residuals[stat_name] = float(pred[0])

        return residuals

    def get_stats(self) -> list[str]:
        """Return list of stat names with trained models."""
        return [name for name, model in self.models.items() if model.is_fitted]

    def get_params(self) -> dict[str, Any]:
        """Return model set state for serialization."""
        return {
            "player_type": self.player_type,
            "models": {name: model.get_params() for name, model in self.models.items()},
            "feature_names": self.feature_names,
            "training_years": self.training_years,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ResidualModelSet:
        """Reconstruct model set from serialized state."""
        model_set = cls(
            player_type=params["player_type"],
            feature_names=params["feature_names"],
            training_years=tuple(params["training_years"]),
        )
        for name, model_params in params["models"].items():
            model_set.models[name] = StatResidualModel.from_params(model_params)
        return model_set
