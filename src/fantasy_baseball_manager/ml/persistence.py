"""Model persistence using joblib for trained ML models.

Delegates to the unified BaseModelStore from the registry package.
Preserves the original public API for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fantasy_baseball_manager.registry.base_store import BaseModelStore
from fantasy_baseball_manager.registry.base_store import ModelMetadata as _RegistryMetadata
from fantasy_baseball_manager.registry.serializers import JoblibSerializer

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.residual_model import ResidualModelSet
    from fantasy_baseball_manager.ml.validation import ValidationReport

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".fantasy_baseball" / "models"


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata about a saved model."""

    name: str
    player_type: str
    training_years: tuple[int, ...]
    stats: list[str]
    feature_names: list[str]
    created_at: str
    validation: dict | None = None


def _to_legacy_metadata(registry_meta: _RegistryMetadata) -> ModelMetadata:
    """Convert registry metadata to the legacy format."""
    return ModelMetadata(
        name=registry_meta.name,
        player_type=registry_meta.player_type,
        training_years=registry_meta.training_years,
        stats=registry_meta.stats,
        feature_names=registry_meta.feature_names,
        created_at=registry_meta.created_at,
        validation=registry_meta.metrics.get("validation"),
    )


@dataclass
class ModelStore:
    """Stores and retrieves trained ML models using joblib.

    Models are saved to ~/.fantasy_baseball/models/ by default.
    Each model has:
    - {name}_{player_type}.joblib: The serialized model set
    - {name}_{player_type}_meta.json: Metadata about the model
    """

    model_dir: Path = DEFAULT_MODEL_DIR
    _store: BaseModelStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = BaseModelStore(
            model_dir=self.model_dir,
            serializer=JoblibSerializer(),
            model_type_name="gb_residual",
        )

    def save(
        self,
        model_set: ResidualModelSet,
        name: str,
        validation_report: ValidationReport | None = None,
        *,
        version: int = 1,
    ) -> Path:
        """Save a model set to disk.

        Args:
            model_set: The trained model set to save
            name: Name for the model (e.g., "default", "2024")
            validation_report: Optional validation report to save with metadata
            version: Version number for this model.

        Returns:
            Path to the saved model file
        """
        validation_dict = validation_report.to_dict() if validation_report else None
        return self._store.save_params(
            model_set.get_params(),
            name,
            model_set.player_type,
            training_years=model_set.training_years,
            stats=model_set.get_stats(),
            feature_names=model_set.feature_names,
            metrics={"validation": validation_dict} if validation_dict else {},
            version=version,
        )

    def load(self, name: str, player_type: str) -> ResidualModelSet:
        """Load a model set from disk.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher"

        Returns:
            The loaded model set

        Raises:
            FileNotFoundError: If the model does not exist
        """
        from fantasy_baseball_manager.ml.residual_model import ResidualModelSet

        params = self._store.load_params(name, player_type)
        return ResidualModelSet.from_params(params)

    def exists(self, name: str, player_type: str) -> bool:
        """Check if a model exists.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher"

        Returns:
            True if the model file exists
        """
        return self._store.exists(name, player_type)

    def get_metadata(self, name: str, player_type: str) -> ModelMetadata | None:
        """Load metadata for a model without loading the full model.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher"

        Returns:
            ModelMetadata if found, None otherwise
        """
        registry_meta = self._store.get_metadata(name, player_type)
        if registry_meta is None:
            return None
        return _to_legacy_metadata(registry_meta)

    def list_models(self) -> list[ModelMetadata]:
        """List all available models.

        Returns:
            List of ModelMetadata for all saved models
        """
        return [_to_legacy_metadata(m) for m in self._store.list_models()]

    def delete(self, name: str, player_type: str) -> bool:
        """Delete a model.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher"

        Returns:
            True if model was deleted, False if it didn't exist
        """
        return self._store.delete(name, player_type)
