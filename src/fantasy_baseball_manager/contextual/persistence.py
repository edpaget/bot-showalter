"""Model persistence for contextual transformer checkpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel

logger = logging.getLogger(__name__)

DEFAULT_CONTEXTUAL_MODEL_DIR = Path.home() / ".fantasy_baseball" / "models" / "contextual"


@dataclass(frozen=True, slots=True)
class ContextualModelMetadata:
    """Metadata about a saved contextual model checkpoint."""

    name: str
    epoch: int
    train_loss: float
    val_loss: float
    pitch_type_accuracy: float | None = None
    pitch_result_accuracy: float | None = None
    created_at: str | None = None


@dataclass
class ContextualModelStore:
    """Stores and retrieves contextual model checkpoints.

    Files per checkpoint:
    - {name}.pt: model state_dict
    - {name}_meta.json: metadata
    - {name}_optimizer.pt: optimizer + scheduler state (optional, for resume)
    """

    model_dir: Path = DEFAULT_CONTEXTUAL_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        name: str,
        model: ContextualPerformanceModel,
        metadata: ContextualModelMetadata,
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
    ) -> Path:
        """Save a model checkpoint to disk."""
        import datetime

        model_path = self._model_path(name)
        meta_path = self._meta_path(name)

        torch.save(model.state_dict(), model_path)
        logger.info("Saved contextual model to %s", model_path)

        created_at = metadata.created_at or datetime.datetime.now(datetime.UTC).isoformat()
        with meta_path.open("w") as f:
            json.dump(
                {
                    "name": metadata.name,
                    "epoch": metadata.epoch,
                    "train_loss": metadata.train_loss,
                    "val_loss": metadata.val_loss,
                    "pitch_type_accuracy": metadata.pitch_type_accuracy,
                    "pitch_result_accuracy": metadata.pitch_result_accuracy,
                    "created_at": created_at,
                },
                f,
                indent=2,
            )

        if optimizer_state is not None:
            opt_path = self._optimizer_path(name)
            torch.save(
                {"optimizer": optimizer_state, "scheduler": scheduler_state},
                opt_path,
            )
            logger.debug("Saved optimizer state to %s", opt_path)

        return model_path

    def load_model(self, name: str, model_config: ModelConfig) -> ContextualPerformanceModel:
        """Load a model from disk.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        from fantasy_baseball_manager.contextual.model.heads import MaskedGamestateHead
        from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel

        model_path = self._model_path(name)
        if not model_path.exists():
            raise FileNotFoundError(f"Contextual model not found: {model_path}")

        model = ContextualPerformanceModel(model_config, MaskedGamestateHead(model_config))
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded contextual model from %s", model_path)
        return model

    def load_training_state(
        self, name: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        """Load model state_dict, optimizer state, and scheduler state for resuming.

        Returns:
            (model_state_dict, optimizer_state, scheduler_state)
        """
        model_path = self._model_path(name)
        if not model_path.exists():
            raise FileNotFoundError(f"Contextual model not found: {model_path}")

        state_dict = torch.load(model_path, weights_only=True)

        opt_path = self._optimizer_path(name)
        optimizer_state = None
        scheduler_state = None
        if opt_path.exists():
            opt_data = torch.load(opt_path, weights_only=False)
            optimizer_state = opt_data["optimizer"]
            scheduler_state = opt_data.get("scheduler")

        return state_dict, optimizer_state, scheduler_state

    def get_metadata(self, name: str) -> ContextualModelMetadata | None:
        """Load metadata for a checkpoint without loading the model."""
        meta_path = self._meta_path(name)
        if not meta_path.exists():
            return None

        with meta_path.open() as f:
            data = json.load(f)
            return ContextualModelMetadata(
                name=data["name"],
                epoch=data["epoch"],
                train_loss=data["train_loss"],
                val_loss=data["val_loss"],
                pitch_type_accuracy=data.get("pitch_type_accuracy"),
                pitch_result_accuracy=data.get("pitch_result_accuracy"),
                created_at=data.get("created_at"),
            )

    def list_checkpoints(self) -> list[ContextualModelMetadata]:
        """List all available checkpoints."""
        checkpoints: list[ContextualModelMetadata] = []
        for meta_path in self.model_dir.glob("*_meta.json"):
            with meta_path.open() as f:
                data = json.load(f)
                checkpoints.append(
                    ContextualModelMetadata(
                        name=data["name"],
                        epoch=data["epoch"],
                        train_loss=data["train_loss"],
                        val_loss=data["val_loss"],
                        pitch_type_accuracy=data.get("pitch_type_accuracy"),
                        pitch_result_accuracy=data.get("pitch_result_accuracy"),
                        created_at=data.get("created_at"),
                    )
                )
        return checkpoints

    def exists(self, name: str) -> bool:
        """Check if a checkpoint exists."""
        return self._model_path(name).exists()

    def delete(self, name: str) -> bool:
        """Delete a checkpoint and its associated files."""
        model_path = self._model_path(name)
        meta_path = self._meta_path(name)
        opt_path = self._optimizer_path(name)

        deleted = False
        if model_path.exists():
            model_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
        if opt_path.exists():
            opt_path.unlink()

        if deleted:
            logger.info("Deleted contextual checkpoint: %s", name)
        return deleted

    def _model_path(self, name: str) -> Path:
        return self.model_dir / f"{name}.pt"

    def _meta_path(self, name: str) -> Path:
        return self.model_dir / f"{name}_meta.json"

    def _optimizer_path(self, name: str) -> Path:
        return self.model_dir / f"{name}_optimizer.pt"
