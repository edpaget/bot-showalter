"""Persistence layer for prepared (tensorized) training data.

Saves and loads pre-built tensorized sequences so that pretrain/finetune
commands can skip the expensive parquet I/O, game-sequence building, and
tensorization steps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedSingle

logger = logging.getLogger(__name__)

DEFAULT_PREPARED_DATA_DIR = (
    Path.home() / ".fantasy_baseball" / "prepared_data" / "contextual"
)


@dataclass
class PreparedDataStore:
    """Stores and retrieves pre-built tensorized training data."""

    data_dir: Path = DEFAULT_PREPARED_DATA_DIR

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_pretrain_data(
        self,
        name: str,
        sequences: list[TensorizedSingle],
        meta: dict[str, Any],
    ) -> None:
        """Save tensorized sequences as {name}.pt and metadata as {name}_meta.json."""
        pt_path = self._pt_path(name)
        torch.save(sequences, pt_path)
        logger.info("Saved %d pretrain sequences to %s", len(sequences), pt_path)
        self._save_meta(name, meta)

    def load_pretrain_data(self, name: str) -> list[TensorizedSingle]:
        """Load list[TensorizedSingle] from {name}.pt."""
        pt_path = self._pt_path(name)
        if not pt_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {pt_path}")
        data: list[TensorizedSingle] = torch.load(pt_path, weights_only=False)
        logger.info("Loaded %d pretrain sequences from %s", len(data), pt_path)
        return data

    def save_finetune_data(
        self,
        name: str,
        windows: list[tuple[TensorizedSingle, torch.Tensor]],
        meta: dict[str, Any],
    ) -> None:
        """Save fine-tune windows as {name}.pt and metadata as {name}_meta.json."""
        pt_path = self._pt_path(name)
        torch.save(windows, pt_path)
        logger.info("Saved %d finetune windows to %s", len(windows), pt_path)
        self._save_meta(name, meta)

    def load_finetune_data(
        self, name: str
    ) -> list[tuple[TensorizedSingle, torch.Tensor]]:
        """Load list[tuple[TensorizedSingle, Tensor]] from {name}.pt."""
        pt_path = self._pt_path(name)
        if not pt_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {pt_path}")
        data: list[tuple[TensorizedSingle, torch.Tensor]] = torch.load(
            pt_path, weights_only=False
        )
        logger.info("Loaded %d finetune windows from %s", len(data), pt_path)
        return data

    def load_meta(self, name: str) -> dict[str, Any] | None:
        """Load {name}_meta.json, or None if missing."""
        meta_path = self._meta_path(name)
        if not meta_path.exists():
            return None
        with meta_path.open() as f:
            result: dict[str, Any] = json.load(f)
            return result

    def exists(self, name: str) -> bool:
        """True if {name}.pt exists."""
        return self._pt_path(name).exists()

    def _save_meta(self, name: str, meta: dict[str, Any]) -> None:
        meta_path = self._meta_path(name)
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

    def _pt_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.pt"

    def _meta_path(self, name: str) -> Path:
        return self.data_dir / f"{name}_meta.json"
