"""Model persistence for MTL PyTorch models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.mtl.model import (
        MultiTaskBatterModel,
        MultiTaskPitcherModel,
    )

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


@dataclass
class MTLModelStore:
    """Stores and retrieves trained MTL models using PyTorch serialization.

    Models are saved to ~/.fantasy_baseball/models/mtl/ by default.
    Each model has:
    - {name}_{player_type}.pt: The PyTorch model state dict
    - {name}_{player_type}_meta.json: Metadata about the model
    """

    model_dir: Path = DEFAULT_MTL_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_batter_model(
        self,
        model: MultiTaskBatterModel,
        name: str,
    ) -> Path:
        """Save a batter model to disk.

        Args:
            model: The trained batter model to save.
            name: Name for the model (e.g., "default", "2024").

        Returns:
            Path to the saved model file.
        """
        return self._save_model(model, name, "batter", model.STATS)

    def save_pitcher_model(
        self,
        model: MultiTaskPitcherModel,
        name: str,
    ) -> Path:
        """Save a pitcher model to disk.

        Args:
            model: The trained pitcher model to save.
            name: Name for the model (e.g., "default", "2024").

        Returns:
            Path to the saved model file.
        """
        return self._save_model(model, name, "pitcher", model.STATS)

    def _save_model(
        self,
        model: MultiTaskBatterModel | MultiTaskPitcherModel,
        name: str,
        player_type: str,
        stats: tuple[str, ...],
    ) -> Path:
        """Save a model to disk."""
        import datetime

        model_path = self._model_path(name, player_type)
        meta_path = self._meta_path(name, player_type)

        # Save model parameters using torch.save
        params = model.get_params()
        torch.save(params, model_path)
        logger.info("Saved MTL %s model to %s", player_type, model_path)

        # Save metadata
        metadata = MTLModelMetadata(
            name=name,
            player_type=player_type,
            training_years=model.training_years,
            stats=list(stats),
            feature_names=model.feature_names,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            validation_metrics=model.validation_metrics,
        )
        with meta_path.open("w") as f:
            json.dump(
                {
                    "name": metadata.name,
                    "player_type": metadata.player_type,
                    "training_years": list(metadata.training_years),
                    "stats": metadata.stats,
                    "feature_names": metadata.feature_names,
                    "created_at": metadata.created_at,
                    "validation_metrics": metadata.validation_metrics,
                },
                f,
                indent=2,
            )
        logger.debug("Saved MTL metadata to %s", meta_path)

        return model_path

    def load_batter_model(self, name: str) -> MultiTaskBatterModel:
        """Load a batter model from disk.

        Args:
            name: Name of the model.

        Returns:
            The loaded batter model.

        Raises:
            FileNotFoundError: If the model does not exist.
        """
        from fantasy_baseball_manager.ml.mtl.model import MultiTaskBatterModel

        model_path = self._model_path(name, "batter")
        if not model_path.exists():
            raise FileNotFoundError(f"MTL batter model not found: {model_path}")

        params = torch.load(model_path, weights_only=False)
        model = MultiTaskBatterModel.from_params(params)
        logger.info("Loaded MTL batter model from %s", model_path)
        return model

    def load_pitcher_model(self, name: str) -> MultiTaskPitcherModel:
        """Load a pitcher model from disk.

        Args:
            name: Name of the model.

        Returns:
            The loaded pitcher model.

        Raises:
            FileNotFoundError: If the model does not exist.
        """
        from fantasy_baseball_manager.ml.mtl.model import MultiTaskPitcherModel

        model_path = self._model_path(name, "pitcher")
        if not model_path.exists():
            raise FileNotFoundError(f"MTL pitcher model not found: {model_path}")

        params = torch.load(model_path, weights_only=False)
        model = MultiTaskPitcherModel.from_params(params)
        logger.info("Loaded MTL pitcher model from %s", model_path)
        return model

    def exists(self, name: str, player_type: str) -> bool:
        """Check if a model exists.

        Args:
            name: Name of the model.
            player_type: "batter" or "pitcher".

        Returns:
            True if the model file exists.
        """
        return self._model_path(name, player_type).exists()

    def get_metadata(self, name: str, player_type: str) -> MTLModelMetadata | None:
        """Load metadata for a model without loading the full model.

        Args:
            name: Name of the model.
            player_type: "batter" or "pitcher".

        Returns:
            MTLModelMetadata if found, None otherwise.
        """
        meta_path = self._meta_path(name, player_type)
        if not meta_path.exists():
            return None

        with meta_path.open() as f:
            data = json.load(f)
            return MTLModelMetadata(
                name=data["name"],
                player_type=data["player_type"],
                training_years=tuple(data["training_years"]),
                stats=data["stats"],
                feature_names=data["feature_names"],
                created_at=data["created_at"],
                validation_metrics=data.get("validation_metrics"),
            )

    def list_models(self) -> list[MTLModelMetadata]:
        """List all available MTL models.

        Returns:
            List of MTLModelMetadata for all saved models.
        """
        models: list[MTLModelMetadata] = []
        for meta_path in self.model_dir.glob("*_meta.json"):
            with meta_path.open() as f:
                data = json.load(f)
                models.append(
                    MTLModelMetadata(
                        name=data["name"],
                        player_type=data["player_type"],
                        training_years=tuple(data["training_years"]),
                        stats=data["stats"],
                        feature_names=data["feature_names"],
                        created_at=data["created_at"],
                        validation_metrics=data.get("validation_metrics"),
                    )
                )
        return models

    def delete(self, name: str, player_type: str) -> bool:
        """Delete a model.

        Args:
            name: Name of the model.
            player_type: "batter" or "pitcher".

        Returns:
            True if model was deleted, False if it didn't exist.
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
            logger.info("Deleted MTL %s model: %s", player_type, name)
        return deleted

    def _model_path(self, name: str, player_type: str) -> Path:
        return self.model_dir / f"{name}_{player_type}.pt"

    def _meta_path(self, name: str, player_type: str) -> Path:
        return self.model_dir / f"{name}_{player_type}_meta.json"
