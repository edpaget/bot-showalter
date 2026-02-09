"""Serialization strategies for model params."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class ModelSerializer(Protocol):
    """Strategy for serializing/deserializing model params to disk."""

    @property
    def extension(self) -> str:
        """File extension including dot (e.g. '.joblib', '.pt')."""
        ...

    def save(self, params: Any, path: Path) -> None:
        """Serialize params to the given path."""
        ...

    def load(self, path: Path) -> Any:
        """Deserialize and return params from the given path."""
        ...


@dataclass(frozen=True)
class JoblibSerializer:
    """Serializer using joblib for scikit-learn/LightGBM models."""

    @property
    def extension(self) -> str:
        return ".joblib"

    def save(self, params: Any, path: Path) -> None:
        import joblib

        joblib.dump(params, path)

    def load(self, path: Path) -> Any:
        import joblib

        return joblib.load(path)


@dataclass(frozen=True)
class TorchParamsSerializer:
    """Serializer using torch.save for PyTorch model params dicts."""

    @property
    def extension(self) -> str:
        return ".pt"

    def save(self, params: Any, path: Path) -> None:
        import torch

        torch.save(params, path)

    def load(self, path: Path) -> Any:
        import torch

        return torch.load(path, weights_only=False)
