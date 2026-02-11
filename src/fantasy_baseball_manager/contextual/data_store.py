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

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedSingle

logger = logging.getLogger(__name__)


def _hierarchical_rows_to_columnar(
    windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> dict[str, torch.Tensor | str]:
    """Convert list-of-tuples hierarchical data to flat columnar dict.

    Context fields are concatenated flat (no padding) with an ``offsets`` tensor
    for O(1) indexing into the flat buffer.  This uses exactly the same memory as
    the original sequences — zero padding waste regardless of max_seq_len.

    Per-window fields (targets, identity, etc.) are stacked normally since they
    have uniform shape.

    Returns:
        Dict with format marker ``"__format__": "columnar_v1"`` and flat/stacked
        tensors.
    """
    n = len(windows)
    if n == 0:
        raise ValueError("Cannot convert empty window list to columnar format")

    # Determine dimensions from first window
    first_ctx = windows[0][0]
    n_numeric = first_ctx.numeric_features.shape[-1]
    n_targets = windows[0][1].shape[0]
    stat_input_dim = windows[0][3].shape[0]

    # Collect seq lengths and compute total tokens / offsets
    seq_lengths = torch.tensor([w[0].seq_length for w in windows], dtype=torch.long)
    total_tokens = int(seq_lengths.sum().item())
    offsets = torch.zeros(n, dtype=torch.long)
    if n > 1:
        offsets[1:] = seq_lengths[:-1].cumsum(0)

    # Pre-allocate flat context fields — zero padding waste.
    # Categorical IDs use int16 (vocab sizes < 100) to save 75% vs int64.
    pitch_type_ids = torch.zeros(total_tokens, dtype=torch.int16)
    pitch_result_ids = torch.zeros(total_tokens, dtype=torch.int16)
    bb_type_ids = torch.zeros(total_tokens, dtype=torch.int16)
    stand_ids = torch.zeros(total_tokens, dtype=torch.int16)
    p_throws_ids = torch.zeros(total_tokens, dtype=torch.int16)
    pa_event_ids = torch.zeros(total_tokens, dtype=torch.int16)
    numeric_features = torch.zeros(total_tokens, n_numeric, dtype=torch.float32)
    numeric_mask = torch.zeros(total_tokens, n_numeric, dtype=torch.bool)
    padding_mask = torch.zeros(total_tokens, dtype=torch.bool)
    player_token_mask = torch.zeros(total_tokens, dtype=torch.bool)
    game_ids = torch.zeros(total_tokens, dtype=torch.int16)

    # Per-window fields (uniform shape, stacked)
    targets = torch.zeros(n, n_targets, dtype=torch.float32)
    context_mean = torch.zeros(n, n_targets, dtype=torch.float32)
    identity_features = torch.zeros(n, stat_input_dim, dtype=torch.float32)
    archetype_ids = torch.zeros(n, dtype=torch.long)

    for i, (ctx, tgt, cm, id_feat, arch_id) in enumerate(windows):
        off = int(offsets[i].item())
        sl = ctx.seq_length
        pitch_type_ids[off:off + sl] = ctx.pitch_type_ids
        pitch_result_ids[off:off + sl] = ctx.pitch_result_ids
        bb_type_ids[off:off + sl] = ctx.bb_type_ids
        stand_ids[off:off + sl] = ctx.stand_ids
        p_throws_ids[off:off + sl] = ctx.p_throws_ids
        pa_event_ids[off:off + sl] = ctx.pa_event_ids
        numeric_features[off:off + sl] = ctx.numeric_features
        numeric_mask[off:off + sl] = ctx.numeric_mask
        padding_mask[off:off + sl] = ctx.padding_mask
        player_token_mask[off:off + sl] = ctx.player_token_mask
        game_ids[off:off + sl] = ctx.game_ids

        targets[i] = tgt
        context_mean[i] = cm
        identity_features[i] = id_feat
        archetype_ids[i] = arch_id

    return {
        "__format__": "columnar_v1",
        "offsets": offsets,
        "seq_lengths": seq_lengths,
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
        columnar = _hierarchical_rows_to_columnar(windows)
        self.save_hierarchical_finetune_columnar(name, columnar, meta)

    def save_hierarchical_finetune_columnar(
        self,
        name: str,
        data: dict[str, torch.Tensor | str],
        meta: dict[str, Any],
    ) -> None:
        """Save a pre-built columnar dict directly as {name}.pt."""
        pt_path = self._pt_path(name)
        torch.save(data, pt_path)
        n = int(data["seq_lengths"].shape[0])  # type: ignore[union-attr]
        logger.info("Saved %d hierarchical finetune windows (columnar) to %s", n, pt_path)
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
        # Try mmap first (columnar format) — avoids loading all data into RAM;
        # the OS pages data in/out as needed.
        raw: object = torch.load(pt_path, weights_only=False, mmap=True)
        if isinstance(raw, dict) and raw.get("__format__") == "columnar_v1":
            n = int(raw["seq_lengths"].shape[0])
            total_tokens = int(raw["seq_lengths"].sum().item())
            logger.info(
                "Loaded %d hierarchical finetune windows (%d total tokens, mmap) from %s",
                n, total_tokens, pt_path,
            )
            return raw
        # Legacy list-of-tuples format — re-load without mmap for conversion
        del raw
        raw_legacy: object = torch.load(pt_path, weights_only=False)
        if isinstance(raw_legacy, list):
            warnings.warn(
                f"Loading legacy row-oriented hierarchical data from {pt_path}. "
                "Re-run 'prepare-data --mode hier-finetune' to save in columnar format.",
                DeprecationWarning,
                stacklevel=2,
            )
            columnar = _hierarchical_rows_to_columnar(raw_legacy)
            del raw_legacy
            logger.info(
                "Loaded and converted %d legacy hierarchical finetune windows from %s",
                columnar["seq_lengths"].shape[0], pt_path,  # type: ignore[union-attr]
            )
            return columnar
        raise TypeError(f"Unexpected data format in {pt_path}: {type(raw_legacy)}")

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
