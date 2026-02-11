"""Dataset utilities for hierarchical model fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from fantasy_baseball_manager.contextual.model.tensorizer import (
    NUMERIC_FIELDS,
    PAD_GAME_ID,
    TensorizedBatch,
    TensorizedSingle,
)
from fantasy_baseball_manager.contextual.training.dataset import (
    _build_player_windows,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.models import PlayerContext
    from fantasy_baseball_manager.contextual.identity.archetypes import ArchetypeModel
    from fantasy_baseball_manager.contextual.identity.stat_profile import (
        PlayerStatProfile,
    )
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
    from fantasy_baseball_manager.contextual.training.config import (
        HierarchicalFineTuneConfig,
    )


@dataclass
class HierarchicalFineTuneSample:
    """A single hierarchical fine-tuning training example."""

    context: TensorizedSingle
    targets: torch.Tensor  # (n_targets,) float
    context_mean: torch.Tensor  # (n_targets,) float
    identity_features: torch.Tensor  # (stat_input_dim,) float
    archetype_id: int


@dataclass
class HierarchicalFineTuneBatch:
    """Collated batch of HierarchicalFineTuneSample instances."""

    context: TensorizedBatch
    targets: torch.Tensor  # (batch, n_targets) float
    context_mean: torch.Tensor  # (batch, n_targets) float
    identity_features: torch.Tensor  # (batch, stat_input_dim) float
    archetype_ids: torch.Tensor  # (batch,) long


def build_hierarchical_windows(
    player_contexts: list[PlayerContext],
    tensorizer: Tensorizer,
    config: HierarchicalFineTuneConfig,
    target_stats: tuple[str, ...],
    profile_lookup: dict[int, PlayerStatProfile],
    archetype_model: ArchetypeModel,
    stat_input_dim: int,
) -> list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Build hierarchical fine-tuning windows with identity features.

    Reuses existing sliding window logic, then attaches identity per player.
    Players without profiles get zero-vector fallback with archetype_id=0.

    Returns:
        List of (tensorized_context, targets, context_mean, identity_features, archetype_id).
    """
    n = config.context_window
    target_mode = config.target_mode
    target_window = config.target_window
    min_required = n + target_window if target_mode == "rates" else n + 1

    eligible = [p for p in player_contexts if len(p.games) >= min_required]

    windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    for player_ctx in eligible:
        # Look up identity
        profile = profile_lookup.get(player_ctx.player_id)
        if profile is not None:
            feat_vec = profile.to_feature_vector()
            identity_features = torch.tensor(feat_vec, dtype=torch.float32)
            archetype_id = int(archetype_model.predict_single(feat_vec))
        else:
            identity_features = torch.zeros(stat_input_dim, dtype=torch.float32)
            archetype_id = 0

        # Build windows using existing logic
        player_windows = _build_player_windows(
            player_ctx,
            tensorizer=tensorizer,
            context_window=n,
            target_stats=target_stats,
            target_mode=target_mode,
            target_window=target_window,
        )
        for tensorized, targets, context_mean in player_windows:
            windows.append((tensorized, targets, context_mean, identity_features, archetype_id))

    return windows


def build_hierarchical_columnar(
    player_contexts: list[PlayerContext],
    tensorizer: Tensorizer,
    config: HierarchicalFineTuneConfig,
    target_stats: tuple[str, ...],
    profile_lookup: dict[int, PlayerStatProfile],
    archetype_model: ArchetypeModel,
    stat_input_dim: int,
) -> dict[str, torch.Tensor | str]:
    """Build hierarchical fine-tuning data directly in flat columnar format.

    Unlike :func:`build_hierarchical_windows`, this never materialises the full
    list-of-tuples.  Per-player windows are iterated and their raw 1-D tensors
    are appended to lightweight Python lists (just pointers).  A single
    ``torch.cat`` at the end produces each contiguous flat buffer.

    Returns:
        Columnar dict suitable for :class:`HierarchicalFineTuneDataset`.
    """
    n = config.context_window
    target_mode = config.target_mode
    target_window = config.target_window
    min_required = n + target_window if target_mode == "rates" else n + 1

    eligible = [p for p in player_contexts if len(p.games) >= min_required]

    # Accumulator lists — only store tensor *references* (~8 bytes each)
    pitch_type_ids_parts: list[torch.Tensor] = []
    pitch_result_ids_parts: list[torch.Tensor] = []
    bb_type_ids_parts: list[torch.Tensor] = []
    stand_ids_parts: list[torch.Tensor] = []
    p_throws_ids_parts: list[torch.Tensor] = []
    pa_event_ids_parts: list[torch.Tensor] = []
    numeric_features_parts: list[torch.Tensor] = []
    numeric_mask_parts: list[torch.Tensor] = []
    padding_mask_parts: list[torch.Tensor] = []
    player_token_mask_parts: list[torch.Tensor] = []
    game_ids_parts: list[torch.Tensor] = []

    seq_lengths_list: list[int] = []
    targets_list: list[torch.Tensor] = []
    context_mean_list: list[torch.Tensor] = []
    identity_features_list: list[torch.Tensor] = []
    archetype_ids_list: list[int] = []

    for player_ctx in eligible:
        # Look up identity (once per player)
        profile = profile_lookup.get(player_ctx.player_id)
        if profile is not None:
            feat_vec = profile.to_feature_vector()
            identity_features = torch.tensor(feat_vec, dtype=torch.float32)
            archetype_id = int(archetype_model.predict_single(feat_vec))
        else:
            identity_features = torch.zeros(stat_input_dim, dtype=torch.float32)
            archetype_id = 0

        # Build windows — yields (TensorizedSingle, targets, context_mean)
        player_windows = _build_player_windows(
            player_ctx,
            tensorizer=tensorizer,
            context_window=n,
            target_stats=target_stats,
            target_mode=target_mode,
            target_window=target_window,
        )
        for tensorized, targets, context_mean in player_windows:
            # Append raw 1-D context tensors (just pointer references)
            pitch_type_ids_parts.append(tensorized.pitch_type_ids)
            pitch_result_ids_parts.append(tensorized.pitch_result_ids)
            bb_type_ids_parts.append(tensorized.bb_type_ids)
            stand_ids_parts.append(tensorized.stand_ids)
            p_throws_ids_parts.append(tensorized.p_throws_ids)
            pa_event_ids_parts.append(tensorized.pa_event_ids)
            numeric_features_parts.append(tensorized.numeric_features)
            numeric_mask_parts.append(tensorized.numeric_mask)
            padding_mask_parts.append(tensorized.padding_mask)
            player_token_mask_parts.append(tensorized.player_token_mask)
            game_ids_parts.append(tensorized.game_ids)

            seq_lengths_list.append(tensorized.seq_length)
            targets_list.append(targets)
            context_mean_list.append(context_mean)
            identity_features_list.append(identity_features)
            archetype_ids_list.append(archetype_id)

    if not seq_lengths_list:
        raise ValueError("No windows produced — check eligibility criteria")

    n_windows = len(seq_lengths_list)
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.long)
    offsets = torch.zeros(n_windows, dtype=torch.long)
    if n_windows > 1:
        offsets[1:] = seq_lengths[:-1].cumsum(0)

    # Categorical IDs stored as int16 (vocab sizes < 100) to save 75% vs int64
    return {
        "__format__": "columnar_v1",
        "offsets": offsets,
        "seq_lengths": seq_lengths,
        "pitch_type_ids": torch.cat(pitch_type_ids_parts).to(torch.int16),
        "pitch_result_ids": torch.cat(pitch_result_ids_parts).to(torch.int16),
        "bb_type_ids": torch.cat(bb_type_ids_parts).to(torch.int16),
        "stand_ids": torch.cat(stand_ids_parts).to(torch.int16),
        "p_throws_ids": torch.cat(p_throws_ids_parts).to(torch.int16),
        "pa_event_ids": torch.cat(pa_event_ids_parts).to(torch.int16),
        "numeric_features": torch.cat(numeric_features_parts),
        "numeric_mask": torch.cat(numeric_mask_parts),
        "padding_mask": torch.cat(padding_mask_parts),
        "player_token_mask": torch.cat(player_token_mask_parts),
        "game_ids": torch.cat(game_ids_parts).to(torch.int16),
        "targets": torch.stack(targets_list),
        "context_mean": torch.stack(context_mean_list),
        "identity_features": torch.stack(identity_features_list),
        "archetype_ids": torch.tensor(archetype_ids_list, dtype=torch.long),
    }


class HierarchicalFineTuneDataset(Dataset[HierarchicalFineTuneSample]):
    """Wraps pre-built hierarchical sliding window examples in columnar format.

    Stores all data as stacked tensors (columnar) for minimal memory overhead.
    Individual samples are sliced out and trimmed to their true sequence length
    on ``__getitem__``.
    """

    def __init__(self, data: dict[str, torch.Tensor | str]) -> None:
        self._data = data

    @classmethod
    def from_windows(
        cls,
        windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor, torch.Tensor, int]],
    ) -> HierarchicalFineTuneDataset:
        """Build a columnar dataset from a list of per-window tuples."""
        from fantasy_baseball_manager.contextual.data_store import (
            _hierarchical_rows_to_columnar,
        )

        return cls(_hierarchical_rows_to_columnar(windows))

    def __len__(self) -> int:
        return int(self._data["seq_lengths"].shape[0])  # type: ignore[union-attr]

    def __getitem__(self, index: int) -> HierarchicalFineTuneSample:
        d = self._data
        off = int(d["offsets"][index].item())  # type: ignore[union-attr]
        sl = int(d["seq_lengths"][index].item())  # type: ignore[union-attr]
        end = off + sl

        # .long() handles both int16 (new format) and int64 (old format)
        context = TensorizedSingle(
            pitch_type_ids=d["pitch_type_ids"][off:end].long(),  # type: ignore[union-attr]
            pitch_result_ids=d["pitch_result_ids"][off:end].long(),  # type: ignore[union-attr]
            bb_type_ids=d["bb_type_ids"][off:end].long(),  # type: ignore[union-attr]
            stand_ids=d["stand_ids"][off:end].long(),  # type: ignore[union-attr]
            p_throws_ids=d["p_throws_ids"][off:end].long(),  # type: ignore[union-attr]
            pa_event_ids=d["pa_event_ids"][off:end].long(),  # type: ignore[union-attr]
            numeric_features=d["numeric_features"][off:end],  # type: ignore[index]
            numeric_mask=d["numeric_mask"][off:end],  # type: ignore[index]
            padding_mask=d["padding_mask"][off:end],  # type: ignore[index]
            player_token_mask=d["player_token_mask"][off:end],  # type: ignore[index]
            game_ids=d["game_ids"][off:end].long(),  # type: ignore[union-attr]
            seq_length=sl,
        )
        return HierarchicalFineTuneSample(
            context=context,
            targets=d["targets"][index],  # type: ignore[index]
            context_mean=d["context_mean"][index],  # type: ignore[index]
            identity_features=d["identity_features"][index],  # type: ignore[index]
            archetype_id=int(d["archetype_ids"][index].item()),  # type: ignore[union-attr]
        )

    def compute_target_std(self) -> torch.Tensor:
        """Per-stat standard deviation of targets, clamped to >= 1e-6."""
        return self._data["targets"].std(dim=0).clamp(min=1e-6)  # type: ignore[union-attr]


def collate_hierarchical_samples(
    samples: list[HierarchicalFineTuneSample],
) -> HierarchicalFineTuneBatch:
    """Pad context fields and stack identity/targets into a HierarchicalFineTuneBatch."""
    max_len = max(s.context.seq_length for s in samples)
    batch_size = len(samples)
    n_numeric = len(NUMERIC_FIELDS)

    pitch_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    pitch_result_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    bb_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    stand_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    p_throws_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    pa_event_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    numeric_features = torch.zeros(batch_size, max_len, n_numeric, dtype=torch.float32)
    numeric_mask = torch.zeros(batch_size, max_len, n_numeric, dtype=torch.bool)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    player_token_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    game_ids = torch.full((batch_size, max_len), PAD_GAME_ID, dtype=torch.long)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, s in enumerate(samples):
        sl = s.context.seq_length
        pitch_type_ids[i, :sl] = s.context.pitch_type_ids
        pitch_result_ids[i, :sl] = s.context.pitch_result_ids
        bb_type_ids[i, :sl] = s.context.bb_type_ids
        stand_ids[i, :sl] = s.context.stand_ids
        p_throws_ids[i, :sl] = s.context.p_throws_ids
        pa_event_ids[i, :sl] = s.context.pa_event_ids
        numeric_features[i, :sl] = s.context.numeric_features
        numeric_mask[i, :sl] = s.context.numeric_mask
        padding_mask[i, :sl] = s.context.padding_mask
        player_token_mask[i, :sl] = s.context.player_token_mask
        game_ids[i, :sl] = s.context.game_ids
        seq_lengths[i] = sl

    context_batch = TensorizedBatch(
        pitch_type_ids=pitch_type_ids,
        pitch_result_ids=pitch_result_ids,
        bb_type_ids=bb_type_ids,
        stand_ids=stand_ids,
        p_throws_ids=p_throws_ids,
        pa_event_ids=pa_event_ids,
        numeric_features=numeric_features,
        numeric_mask=numeric_mask,
        padding_mask=padding_mask,
        player_token_mask=player_token_mask,
        game_ids=game_ids,
        seq_lengths=seq_lengths,
    )

    targets = torch.stack([s.targets for s in samples])
    context_mean = torch.stack([s.context_mean for s in samples])
    identity_features = torch.stack([s.identity_features for s in samples])
    archetype_ids = torch.tensor([s.archetype_id for s in samples], dtype=torch.long)

    return HierarchicalFineTuneBatch(
        context=context_batch,
        targets=targets,
        context_mean=context_mean,
        identity_features=identity_features,
        archetype_ids=archetype_ids,
    )
