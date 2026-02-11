"""Player archetype clustering.

Groups players into archetypes based on their stat profiles
using KMeans clustering on standardized feature vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.identity.stat_profile import PlayerStatProfile

DEFAULT_ARCHETYPE_MODEL_DIR: Path = (
    Path.home() / ".fantasy_baseball" / "models" / "archetypes"
)


@dataclass
class ArchetypeModel:
    """KMeans archetype model with StandardScaler preprocessing.

    Clusters player feature vectors into archetypes. Features are
    standardized before clustering so all dimensions contribute equally.
    """

    n_archetypes: int
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _kmeans: KMeans | None = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: np.ndarray) -> None:
        """Fit scaler and KMeans on feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
        """
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._kmeans = KMeans(
            n_clusters=self.n_archetypes, random_state=42, n_init=10
        )
        self._kmeans.fit(X_scaled)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict archetype labels for feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of integer labels, shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self._is_fitted or self._scaler is None or self._kmeans is None:
            msg = "Model must be fitted before calling predict."
            raise ValueError(msg)
        X_scaled = self._scaler.transform(X)
        return self._kmeans.predict(X_scaled)

    def predict_single(self, x: np.ndarray) -> int:
        """Predict archetype label for a single feature vector.

        Args:
            x: Feature vector of shape (n_features,).

        Returns:
            Integer archetype label.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self._is_fitted:
            msg = "Model must be fitted before calling predict_single."
            raise ValueError(msg)
        labels = self.predict(x.reshape(1, -1))
        return int(labels[0])

    def centroids(self) -> np.ndarray:
        """Return cluster centroids in original (unscaled) feature space.

        Returns:
            Array of shape (n_archetypes, n_features).

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self._is_fitted or self._scaler is None or self._kmeans is None:
            msg = "Model must be fitted before accessing centroids."
            raise ValueError(msg)
        return self._scaler.inverse_transform(self._kmeans.cluster_centers_)

    def get_params(self) -> dict[str, Any]:
        """Serialize model state to a dictionary."""
        return {
            "n_archetypes": self.n_archetypes,
            "scaler": self._scaler,
            "kmeans": self._kmeans,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ArchetypeModel:
        """Reconstruct model from serialized parameters."""
        model = cls(n_archetypes=params["n_archetypes"])
        model._scaler = params.get("scaler")
        model._kmeans = params.get("kmeans")
        model._is_fitted = params.get("is_fitted", False)
        return model


def save_archetype_model(
    model: ArchetypeModel,
    name: str,
    directory: Path | None = None,
) -> Path:
    """Save archetype model to disk as joblib + JSON metadata sidecar.

    Returns the directory where files were saved.
    """
    base = directory or DEFAULT_ARCHETYPE_MODEL_DIR
    model_dir = base / name
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model.get_params(), model_dir / "model.joblib")

    metadata = {
        "n_archetypes": model.n_archetypes,
        "is_fitted": model.is_fitted,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_dir


def load_archetype_model(
    name: str,
    directory: Path | None = None,
) -> ArchetypeModel:
    """Load archetype model from disk.

    Raises:
        FileNotFoundError: If model files do not exist.
    """
    base = directory or DEFAULT_ARCHETYPE_MODEL_DIR
    model_path = base / name / "model.joblib"
    if not model_path.exists():
        msg = f"No archetype model found at {model_path}"
        raise FileNotFoundError(msg)
    params = joblib.load(model_path)
    return ArchetypeModel.from_params(params)


def fit_archetypes(
    profiles: list[PlayerStatProfile],
    n_archetypes: int = 8,
) -> tuple[ArchetypeModel, np.ndarray]:
    """Fit archetype model from player profiles.

    Args:
        profiles: List of PlayerStatProfile instances (must all be same player_type).
        n_archetypes: Number of archetypes to cluster into.

    Returns:
        Tuple of (fitted ArchetypeModel, label array).

    Raises:
        ValueError: If profiles contain mixed player_types.
    """
    if not profiles:
        msg = "Cannot fit archetypes with no profiles."
        raise ValueError(msg)

    player_types = {p.player_type for p in profiles}
    if len(player_types) > 1:
        msg = f"All profiles must have the same player_type, got: {player_types}"
        raise ValueError(msg)

    # Cap n_archetypes to number of profiles
    effective_n = min(n_archetypes, len(profiles))

    X = np.array([p.to_feature_vector() for p in profiles])
    model = ArchetypeModel(n_archetypes=effective_n)
    model.fit(X)
    labels = model.predict(X)
    return model, labels
