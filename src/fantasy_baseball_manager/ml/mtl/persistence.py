"""Model persistence for MTL PyTorch models.

Delegates to the unified MTLBaseModelStore from the registry package.
Preserves the original public API for backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore
from fantasy_baseball_manager.registry.serializers import TorchParamsSerializer

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.mtl.model import (
        MultiTaskBatterModel,
        MultiTaskPitcherModel,
    )
    from fantasy_baseball_manager.registry.base_store import ModelMetadata as _RegistryMetadata

logger = logging.getLogger(__name__)

DEFAULT_MTL_MODEL_DIR = Path.home() / ".fantasy_baseball" / "models" / "mtl"


@dataclass(frozen=True)
class MTLModelMetadata:
    """Metadata about a saved MTL model."""

    name: str
    player_type: str
    training_years: tuple[int, ...]
    stats: list[str]
    feature_names: list[str]
    created_at: str
    validation_metrics: dict[str, float] | None = None


def _to_legacy_metadata(registry_meta: _RegistryMetadata) -> MTLModelMetadata:
    """Convert registry metadata to the legacy format."""
    return MTLModelMetadata(
        name=registry_meta.name,
        player_type=registry_meta.player_type,
        training_years=registry_meta.training_years,
        stats=registry_meta.stats,
        feature_names=registry_meta.feature_names,
        created_at=registry_meta.created_at,
        validation_metrics=registry_meta.metrics.get("validation_metrics"),
    )


@dataclass
class MTLModelStore:
    """Stores and retrieves trained MTL models using PyTorch serialization.

    Models are saved to ~/.fantasy_baseball/models/mtl/ by default.
    Each model has:
    - {name}_{player_type}.pt: The PyTorch model state dict
    - {name}_{player_type}_meta.json: Metadata about the model
    """

    model_dir: Path = DEFAULT_MTL_MODEL_DIR
    _store: MTLBaseModelStore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._store = MTLBaseModelStore(
            model_dir=self.model_dir,
            serializer=TorchParamsSerializer(),
            model_type_name="mtl",
        )

    def save_batter_model(
        self,
        model: MultiTaskBatterModel,
        name: str,
        *,
        version: int = 1,
    ) -> Path:
        """Save a batter model to disk.

        Args:
            model: The trained batter model to save.
            name: Name for the model (e.g., "default", "2024").
            version: Version number for this model.

        Returns:
            Path to the saved model file.
        """
        return self._store.save_model(model, name, "batter", version=version)

    def save_pitcher_model(
        self,
        model: MultiTaskPitcherModel,
        name: str,
        *,
        version: int = 1,
    ) -> Path:
        """Save a pitcher model to disk.

        Args:
            model: The trained pitcher model to save.
            name: Name for the model (e.g., "default", "2024").
            version: Version number for this model.

        Returns:
            Path to the saved model file.
        """
        return self._store.save_model(model, name, "pitcher", version=version)

    def load_batter_model(self, name: str) -> MultiTaskBatterModel:
        """Load a batter model from disk.

        Args:
            name: Name of the model.

        Returns:
            The loaded batter model.

        Raises:
            FileNotFoundError: If the model does not exist.
        """
        return self._store.load_batter(name)

    def load_pitcher_model(self, name: str) -> MultiTaskPitcherModel:
        """Load a pitcher model from disk.

        Args:
            name: Name of the model.

        Returns:
            The loaded pitcher model.

        Raises:
            FileNotFoundError: If the model does not exist.
        """
        return self._store.load_pitcher(name)

    def exists(self, name: str, player_type: str) -> bool:
        """Check if a model exists.

        Args:
            name: Name of the model.
            player_type: "batter" or "pitcher".

        Returns:
            True if the model file exists.
        """
        return self._store.exists(name, player_type)

    def get_metadata(self, name: str, player_type: str) -> MTLModelMetadata | None:
        """Load metadata for a model without loading the full model.

        Args:
            name: Name of the model.
            player_type: "batter" or "pitcher".

        Returns:
            MTLModelMetadata if found, None otherwise.
        """
        registry_meta = self._store.get_metadata(name, player_type)
        if registry_meta is None:
            return None
        return _to_legacy_metadata(registry_meta)

    def list_models(self) -> list[MTLModelMetadata]:
        """List all available MTL models.

        Returns:
            List of MTLModelMetadata for all saved models.
        """
        return [_to_legacy_metadata(m) for m in self._store.list_models()]

    def delete(self, name: str, player_type: str) -> bool:
        """Delete a model.

        Args:
            name: Name of the model.
            player_type: "batter" or "pitcher".

        Returns:
            True if model was deleted, False if it didn't exist.
        """
        return self._store.delete(name, player_type)
