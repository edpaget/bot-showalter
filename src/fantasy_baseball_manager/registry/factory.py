"""Factory functions for creating model stores and the registry."""

from __future__ import annotations

from pathlib import Path

from fantasy_baseball_manager.registry.base_store import BaseModelStore
from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore
from fantasy_baseball_manager.registry.registry import ModelRegistry
from fantasy_baseball_manager.registry.serializers import JoblibSerializer, TorchParamsSerializer

DEFAULT_MODEL_DIR = Path.home() / ".fantasy_baseball" / "models"


def create_gb_store(base_dir: Path = DEFAULT_MODEL_DIR) -> BaseModelStore:
    """Create a store for gradient boosting residual models."""
    return BaseModelStore(
        model_dir=base_dir,
        serializer=JoblibSerializer(),
        model_type_name="gb_residual",
    )


def create_mtl_store(base_dir: Path = DEFAULT_MODEL_DIR) -> MTLBaseModelStore:
    """Create a store for multi-task learning models."""
    return MTLBaseModelStore(
        model_dir=base_dir / "mtl",
        serializer=TorchParamsSerializer(),
        model_type_name="mtl",
    )


def create_mle_store(base_dir: Path = DEFAULT_MODEL_DIR) -> BaseModelStore:
    """Create a store for minor league equivalency models."""
    return BaseModelStore(
        model_dir=base_dir / "mle",
        serializer=JoblibSerializer(),
        model_type_name="mle",
    )


def create_model_registry(base_dir: Path = DEFAULT_MODEL_DIR) -> ModelRegistry:
    """Create a fully-wired model registry with all stores.

    Args:
        base_dir: Root directory for all model storage.
            Defaults to ~/.fantasy_baseball/models.
    """
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore

    return ModelRegistry(
        gb_store=create_gb_store(base_dir),
        mtl_store=create_mtl_store(base_dir),
        mle_store=create_mle_store(base_dir),
        contextual_store=ContextualModelStore(model_dir=base_dir / "contextual"),
    )
