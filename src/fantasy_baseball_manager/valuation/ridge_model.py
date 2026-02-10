"""Ridge regression valuation model.

Wraps sklearn Ridge + StandardScaler in a dataclass with fit/predict/
get_params/from_params and file-based persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import numpy as np

DEFAULT_VALUATION_MODEL_DIR: Path = (
    Path.home() / ".fantasy_baseball" / "models" / "valuation"
)


@dataclass(frozen=True)
class RidgeValuationConfig:
    """Configuration for ridge valuation training."""

    alpha: float = 1.0
    target_transform: str = "log"  # "log" or "raw"
    min_pa: int = 50
    min_ip: float = 20.0


@dataclass
class RidgeValuationModel:
    """Ridge regression model that predicts log(ADP) from stat features."""

    player_type: str  # "batter" or "pitcher"
    config: RidgeValuationConfig = field(default_factory=RidgeValuationConfig)
    feature_names: tuple[str, ...] = ()
    training_years: tuple[int, ...] = ()
    validation_metrics: dict[str, float] = field(default_factory=dict)
    _model: Ridge | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: tuple[str, ...],
    ) -> None:
        """Fit StandardScaler + Ridge on training data."""
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = Ridge(alpha=self.config.alpha)
        self._model.fit(X_scaled, y)
        self.feature_names = feature_names
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict log(ADP) from features.

        Returns raw model output (log-scale ADP prediction).

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self._is_fitted or self._scaler is None or self._model is None:
            msg = "Model must be fitted before calling predict."
            raise ValueError(msg)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_value(self, X: np.ndarray) -> np.ndarray:
        """Predict fantasy value (higher = better).

        Negates log(ADP) so that lower ADP = higher value, matching
        the PlayerValue convention where higher total_value is better.
        """
        return -self.predict(X)

    def coefficients(self) -> dict[str, float]:
        """Return ridge coefficients mapped to feature names.

        Coefficients are on the *scaled* features so they are comparable
        across categories.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self._is_fitted or self._model is None:
            msg = "Model must be fitted before accessing coefficients."
            raise ValueError(msg)
        return dict(zip(self.feature_names, self._model.coef_, strict=True))

    def get_params(self) -> dict[str, Any]:
        """Serialize model state to a dictionary."""
        return {
            "player_type": self.player_type,
            "config": {
                "alpha": self.config.alpha,
                "target_transform": self.config.target_transform,
                "min_pa": self.config.min_pa,
                "min_ip": self.config.min_ip,
            },
            "feature_names": list(self.feature_names),
            "training_years": list(self.training_years),
            "validation_metrics": dict(self.validation_metrics),
            "model": self._model,
            "scaler": self._scaler,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RidgeValuationModel:
        """Reconstruct model from serialized parameters."""
        config = RidgeValuationConfig(**params["config"])
        model = cls(
            player_type=params["player_type"],
            config=config,
            feature_names=tuple(params["feature_names"]),
            training_years=tuple(params["training_years"]),
            validation_metrics=dict(params.get("validation_metrics", {})),
        )
        model._model = params.get("model")
        model._scaler = params.get("scaler")
        model._is_fitted = params.get("is_fitted", False)
        return model


def save_model(
    model: RidgeValuationModel,
    name: str,
    directory: Path | None = None,
) -> Path:
    """Save model to disk as joblib + JSON metadata sidecar.

    Returns the directory where files were saved.
    """
    base = directory or DEFAULT_VALUATION_MODEL_DIR
    model_dir = base / name / model.player_type
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save sklearn objects via joblib
    joblib.dump(model.get_params(), model_dir / "model.joblib")

    # Save human-readable metadata sidecar
    metadata = {
        "player_type": model.player_type,
        "feature_names": list(model.feature_names),
        "training_years": list(model.training_years),
        "validation_metrics": model.validation_metrics,
        "config": {
            "alpha": model.config.alpha,
            "target_transform": model.config.target_transform,
            "min_pa": model.config.min_pa,
            "min_ip": model.config.min_ip,
        },
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir


def load_model(
    name: str,
    player_type: str,
    directory: Path | None = None,
) -> RidgeValuationModel:
    """Load model from disk.

    Raises:
        FileNotFoundError: If model files do not exist.
    """
    base = directory or DEFAULT_VALUATION_MODEL_DIR
    model_path = base / name / player_type / "model.joblib"
    if not model_path.exists():
        msg = f"No valuation model found at {model_path}"
        raise FileNotFoundError(msg)
    params = joblib.load(model_path)
    return RidgeValuationModel.from_params(params)
