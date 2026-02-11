"""Persistence layer for prepared (tensorized) training data.

Saves and loads pre-built tensorized sequences so that pretrain/finetune
commands can skip the expensive parquet I/O, game-sequence building, and
tensorization steps.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from fantasy_baseball_manager.contextual.model.tensorizer import PAD_GAME_ID

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedSingle

logger = logging.getLogger(__name__)


def _hierarchical_rows_to_columnar(
    windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> dict[str, torch.Tensor | str]:
    """Convert list-of-tuples hierarchical data to columnar dict of stacked tensors.

    Pre-allocates stacked tensors at global max sequence length, then copies each
    window's data via slice assignment.

    Returns:
        Dict with format marker ``"__format__": "columnar_v1"`` and stacked tensors.
    """
    n = len(windows)
    if n == 0:
        raise ValueError("Cannot convert empty window list to columnar format")

    # Determine dimensions from first window
    first_ctx = windows[0][0]
    n_numeric = first_ctx.numeric_features.shape[-1]
    n_targets = windows[0][1].shape[0]
    stat_input_dim = windows[0][3].shape[0]

    # Find global max sequence length
    max_seq = max(w[0].seq_length for w in windows)

    # Pre-allocate context fields (padded)
    pitch_type_ids = torch.zeros(n, max_seq, dtype=torch.long)
    pitch_result_ids = torch.zeros(n, max_seq, dtype=torch.long)
    bb_type_ids = torch.zeros(n, max_seq, dtype=torch.long)
    stand_ids = torch.zeros(n, max_seq, dtype=torch.long)
    p_throws_ids = torch.zeros(n, max_seq, dtype=torch.long)
    pa_event_ids = torch.zeros(n, max_seq, dtype=torch.long)
    numeric_features = torch.zeros(n, max_seq, n_numeric, dtype=torch.float32)
    numeric_mask = torch.zeros(n, max_seq, n_numeric, dtype=torch.bool)
    padding_mask = torch.zeros(n, max_seq, dtype=torch.bool)
    player_token_mask = torch.zeros(n, max_seq, dtype=torch.bool)
    game_ids = torch.full((n, max_seq), PAD_GAME_ID, dtype=torch.long)
    seq_lengths = torch.zeros(n, dtype=torch.long)

    # Per-window fields
    targets = torch.zeros(n, n_targets, dtype=torch.float32)
    context_mean = torch.zeros(n, n_targets, dtype=torch.float32)
    identity_features = torch.zeros(n, stat_input_dim, dtype=torch.float32)
    archetype_ids = torch.zeros(n, dtype=torch.long)

    for i, (ctx, tgt, cm, id_feat, arch_id) in enumerate(windows):
        sl = ctx.seq_length
        pitch_type_ids[i, :sl] = ctx.pitch_type_ids
        pitch_result_ids[i, :sl] = ctx.pitch_result_ids
        bb_type_ids[i, :sl] = ctx.bb_type_ids
        stand_ids[i, :sl] = ctx.stand_ids
        p_throws_ids[i, :sl] = ctx.p_throws_ids
        pa_event_ids[i, :sl] = ctx.pa_event_ids
        numeric_features[i, :sl] = ctx.numeric_features
        numeric_mask[i, :sl] = ctx.numeric_mask
        padding_mask[i, :sl] = ctx.padding_mask
        player_token_mask[i, :sl] = ctx.player_token_mask
        game_ids[i, :sl] = ctx.game_ids
        seq_lengths[i] = sl

        targets[i] = tgt
        context_mean[i] = cm
        identity_features[i] = id_feat
        archetype_ids[i] = arch_id

    return {
        "__format__": "columnar_v1",
        "pitch_type_ids": pitch_type_ids,
        "pitch_result_ids": pitch_result_ids,
        "bb_type_ids": bb_type_ids,
        "stand_ids": stand_ids,
        "p_throws_ids": p_throws_ids,
        "pa_event_ids": pa_event_ids,
        "numeric_features": numeric_features,
        "numeric_mask": numeric_mask,
        "padding_mask": padding_mask,
        "player_token_mask": player_token_mask,
        "game_ids": game_ids,
        "seq_lengths": seq_lengths,
        "targets": targets,
        "context_mean": context_mean,
        "identity_features": identity_features,
        "archetype_ids": archetype_ids,
    }

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
        windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]],
        meta: dict[str, Any],
    ) -> None:
        """Save fine-tune windows as {name}.pt and metadata as {name}_meta.json."""
        pt_path = self._pt_path(name)
        torch.save(windows, pt_path)
        logger.info("Saved %d finetune windows to %s", len(windows), pt_path)
        self._save_meta(name, meta)

    def load_finetune_data(
        self, name: str
    ) -> list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]]:
        """Load list[tuple[TensorizedSingle, Tensor, Tensor]] from {name}.pt."""
        pt_path = self._pt_path(name)
        if not pt_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {pt_path}")
        data: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]] = torch.load(
            pt_path, weights_only=False
        )
        logger.info("Loaded %d finetune windows from %s", len(data), pt_path)
        return data

    def save_hierarchical_finetune_data(
        self,
        name: str,
        windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor, torch.Tensor, int]],
        meta: dict[str, Any],
    ) -> None:
        """Save hierarchical fine-tune windows as columnar tensors in {name}.pt."""
        pt_path = self._pt_path(name)
        columnar = _hierarchical_rows_to_columnar(windows)
        torch.save(columnar, pt_path)
        logger.info("Saved %d hierarchical finetune windows (columnar) to %s", len(windows), pt_path)
        self._save_meta(name, meta)

    def load_hierarchical_finetune_data(
        self, name: str
    ) -> dict[str, torch.Tensor | str]:
        """Load hierarchical fine-tune data from {name}.pt as columnar dict.

        Detects legacy list-of-tuples format and converts on-the-fly with a
        deprecation warning.
        """
        pt_path = self._pt_path(name)
        if not pt_path.exists():
            raise FileNotFoundError(f"Prepared data not found: {pt_path}")
        raw: object = torch.load(pt_path, weights_only=False)
        if isinstance(raw, dict) and raw.get("__format__") == "columnar_v1":
            n = int(raw["seq_lengths"].shape[0])
            logger.info("Loaded %d hierarchical finetune windows (columnar) from %s", n, pt_path)
            return raw
        # Legacy list-of-tuples format
        if isinstance(raw, list):
            warnings.warn(
                f"Loading legacy row-oriented hierarchical data from {pt_path}. "
                "Re-run 'prepare-data --mode hier-finetune' to save in columnar format.",
                DeprecationWarning,
                stacklevel=2,
            )
            columnar = _hierarchical_rows_to_columnar(raw)
            logger.info(
                "Loaded and converted %d legacy hierarchical finetune windows from %s",
                len(raw), pt_path,
            )
            return columnar
        raise TypeError(f"Unexpected data format in {pt_path}: {type(raw)}")

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
