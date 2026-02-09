"""Generic model store handling file I/O, metadata, and listing."""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from fantasy_baseball_manager.registry.serializers import ModelSerializer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata about a saved model, stored as a JSON sidecar."""

    name: str
    model_type: str
    player_type: str
    version: int
    training_years: tuple[int, ...]
    stats: list[str]
    feature_names: list[str]
    created_at: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "player_type": self.player_type,
            "version": self.version,
            "training_years": list(self.training_years),
            "stats": self.stats,
            "feature_names": self.feature_names,
            "created_at": self.created_at,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Parse metadata from a JSON dict, handling legacy formats."""
        # Legacy files may lack model_type/version; provide defaults
        return cls(
            name=data["name"],
            model_type=data.get("model_type", "unknown"),
            player_type=data.get("player_type", ""),
            version=data.get("version", 1),
            training_years=tuple(data.get("training_years", ())),
            stats=data.get("stats", []),
            feature_names=data.get("feature_names", []),
            created_at=data.get("created_at", ""),
            metrics=data.get("metrics", {}),
        )


@dataclass
class BaseModelStore:
    """Low-level model store handling serialization, metadata, and file management.

    This class handles the common plumbing shared across GB, MTL, and MLE
    model stores: file path conventions, JSON sidecar metadata, listing,
    and deletion. Model-specific logic (calling get_params/from_params,
    extracting metadata fields) remains in the wrapper classes.
    """

    model_dir: Path
    serializer: ModelSerializer
    model_type_name: str

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_params(
        self,
        params: Any,
        name: str,
        player_type: str,
        *,
        training_years: tuple[int, ...] = (),
        stats: list[str] | None = None,
        feature_names: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        version: int = 1,
    ) -> Path:
        """Serialize params and write metadata JSON sidecar.

        Args:
            params: Serializable model parameters (from model.get_params()).
            name: Model name (e.g. "default", "default_v2").
            player_type: "batter" or "pitcher".
            training_years: Years used for training data.
            stats: Stat names this model covers.
            feature_names: Input feature names.
            metrics: Validation metrics or other evaluation data.
            version: Version number for this model.

        Returns:
            Path to the saved model file.
        """
        model_path = self._model_path(name, player_type)
        meta_path = self._meta_path(name, player_type)

        self.serializer.save(params, model_path)
        logger.info("Saved %s %s model to %s", self.model_type_name, player_type, model_path)

        metadata = ModelMetadata(
            name=name,
            model_type=self.model_type_name,
            player_type=player_type,
            version=version,
            training_years=training_years,
            stats=stats or [],
            feature_names=feature_names or [],
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            metrics=metrics or {},
        )
        with meta_path.open("w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.debug("Saved metadata to %s", meta_path)

        return model_path

    def load_params(self, name: str, player_type: str) -> Any:
        """Deserialize and return raw model params.

        Args:
            name: Model name.
            player_type: "batter" or "pitcher".

        Returns:
            Deserialized params dict/object.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = self._model_path(name, player_type)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        params = self.serializer.load(model_path)
        logger.info("Loaded %s %s model from %s", self.model_type_name, player_type, model_path)
        return params

    def exists(self, name: str, player_type: str) -> bool:
        """Check if a model exists on disk."""
        return self._model_path(name, player_type).exists()

    def get_metadata(self, name: str, player_type: str) -> ModelMetadata | None:
        """Load metadata without loading the model.

        Returns None if the metadata file doesn't exist.
        """
        meta_path = self._meta_path(name, player_type)
        if not meta_path.exists():
            return None

        with meta_path.open() as f:
            data = json.load(f)
            return ModelMetadata.from_dict(data)

    def list_models(self) -> list[ModelMetadata]:
        """List all models in this store by scanning metadata files."""
        models: list[ModelMetadata] = []
        for meta_path in sorted(self.model_dir.glob("*_meta.json")):
            with meta_path.open() as f:
                data = json.load(f)
                models.append(ModelMetadata.from_dict(data))
        return models

    def delete(self, name: str, player_type: str) -> bool:
        """Delete a model and its metadata.

        Returns True if the model was deleted, False if it didn't exist.
        """
        model_path = self._model_path(name, player_type)
        meta_path = self._meta_path(name, player_type)

        deleted = False
        if model_path.exists():
            model_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()

        if deleted:
            logger.info("Deleted %s %s model: %s", self.model_type_name, player_type, name)
        return deleted

    def _model_path(self, name: str, player_type: str) -> Path:
        return self.model_dir / f"{name}_{player_type}{self.serializer.extension}"

    def _meta_path(self, name: str, player_type: str) -> Path:
        return self.model_dir / f"{name}_{player_type}_meta.json"
