"""Model persistence using joblib for trained MLE models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.minors.model import MLEGradientBoostingModel

logger = logging.getLogger(__name__)

DEFAULT_MLE_MODEL_DIR = Path.home() / ".fantasy_baseball" / "models" / "mle"


@dataclass(frozen=True)
class MLEModelMetadata:
    """Metadata about a saved MLE model."""

    name: str
    player_type: str
    training_years: tuple[int, ...]
    stats: list[str]
    feature_names: list[str]
    created_at: str
    validation_metrics: dict | None = None


@dataclass
class MLEModelStore:
    """Stores and retrieves trained MLE models using joblib.

    Models are saved to ~/.fantasy_baseball/models/mle/ by default.
    Each model has:
    - {name}_{player_type}.joblib: The serialized model set
    - {name}_{player_type}_meta.json: Metadata about the model
    """

    model_dir: Path = DEFAULT_MLE_MODEL_DIR

    def __post_init__(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_set: MLEGradientBoostingModel,
        name: str,
        validation_metrics: dict | None = None,
    ) -> Path:
        """Save a model set to disk.

        Args:
            model_set: The trained model set to save
            name: Name for the model (e.g., "default", "2024")
            validation_metrics: Optional validation metrics to save with metadata

        Returns:
            Path to the saved model file
        """
        import datetime

        import joblib

        player_type = model_set.player_type
        model_path = self._model_path(name, player_type)
        meta_path = self._meta_path(name, player_type)

        # Save the model
        params = model_set.get_params()
        joblib.dump(params, model_path)
        logger.info("Saved MLE %s model to %s", player_type, model_path)

        # Save metadata
        metadata = MLEModelMetadata(
            name=name,
            player_type=player_type,
            training_years=model_set.training_years,
            stats=model_set.get_stats(),
            feature_names=model_set.feature_names,
            created_at=datetime.datetime.now(datetime.UTC).isoformat(),
            validation_metrics=validation_metrics,
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
        logger.debug("Saved metadata to %s", meta_path)

        return model_path

    def load(self, name: str, player_type: str = "batter") -> MLEGradientBoostingModel:
        """Load a model set from disk.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher" (default: "batter")

        Returns:
            The loaded model set

        Raises:
            FileNotFoundError: If the model does not exist
        """
        import joblib

        from fantasy_baseball_manager.minors.model import MLEGradientBoostingModel

        model_path = self._model_path(name, player_type)
        if not model_path.exists():
            raise FileNotFoundError(f"MLE model not found: {model_path}")

        params = joblib.load(model_path)
        model_set = MLEGradientBoostingModel.from_params(params)
        logger.info("Loaded MLE %s model from %s", player_type, model_path)
        return model_set

    def exists(self, name: str, player_type: str = "batter") -> bool:
        """Check if a model exists.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher" (default: "batter")

        Returns:
            True if the model file exists
        """
        return self._model_path(name, player_type).exists()

    def get_metadata(self, name: str, player_type: str = "batter") -> MLEModelMetadata | None:
        """Load metadata for a model without loading the full model.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher" (default: "batter")

        Returns:
            MLEModelMetadata if found, None otherwise
        """
        meta_path = self._meta_path(name, player_type)
        if not meta_path.exists():
            return None

        with meta_path.open() as f:
            data = json.load(f)
            return MLEModelMetadata(
                name=data["name"],
                player_type=data["player_type"],
                training_years=tuple(data["training_years"]),
                stats=data["stats"],
                feature_names=data["feature_names"],
                created_at=data["created_at"],
                validation_metrics=data.get("validation_metrics"),
            )

    def list_models(self) -> list[MLEModelMetadata]:
        """List all available MLE models.

        Returns:
            List of MLEModelMetadata for all saved models
        """
        models: list[MLEModelMetadata] = []
        for meta_path in self.model_dir.glob("*_meta.json"):
            with meta_path.open() as f:
                data = json.load(f)
                models.append(
                    MLEModelMetadata(
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

    def delete(self, name: str, player_type: str = "batter") -> bool:
        """Delete a model.

        Args:
            name: Name of the model
            player_type: "batter" or "pitcher" (default: "batter")

        Returns:
            True if model was deleted, False if it didn't exist
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
            logger.info("Deleted MLE %s model: %s", player_type, name)
        return deleted

    def _model_path(self, name: str, player_type: str) -> Path:
        return self.model_dir / f"{name}_{player_type}.joblib"

    def _meta_path(self, name: str, player_type: str) -> Path:
        return self.model_dir / f"{name}_{player_type}_meta.json"
