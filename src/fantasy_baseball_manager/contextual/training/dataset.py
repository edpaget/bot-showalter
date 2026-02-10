"""MGM dataset with BERT-style masking for pre-training, and fine-tune dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from fantasy_baseball_manager.contextual.model.tensorizer import (
    NUMERIC_FIELDS,
    TensorizedBatch,
    TensorizedSingle,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.models import GameSequence, PlayerContext
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
    from fantasy_baseball_manager.contextual.training.config import FineTuneConfig, PreTrainingConfig


@dataclass
class MaskedSample:
    """A single tensorized sequence with masking applied."""

    pitch_type_ids: torch.Tensor  # (seq_len,) long — masked inputs
    pitch_result_ids: torch.Tensor  # (seq_len,) long — masked inputs
    bb_type_ids: torch.Tensor  # (seq_len,) long
    stand_ids: torch.Tensor  # (seq_len,) long
    p_throws_ids: torch.Tensor  # (seq_len,) long
    pa_event_ids: torch.Tensor  # (seq_len,) long
    numeric_features: torch.Tensor  # (seq_len, 23) float
    numeric_mask: torch.Tensor  # (seq_len, 23) bool
    padding_mask: torch.Tensor  # (seq_len,) bool — True=real
    player_token_mask: torch.Tensor  # (seq_len,) bool — True=player slot
    game_ids: torch.Tensor  # (seq_len,) long
    target_pitch_type_ids: torch.Tensor  # (seq_len,) long — originals at masked pos, 0 elsewhere
    target_pitch_result_ids: torch.Tensor  # (seq_len,) long — originals at masked pos, 0 elsewhere
    mask_positions: torch.Tensor  # (seq_len,) bool — True where masked
    seq_length: int = field(default=0)


@dataclass
class MaskedBatch:
    """Collated batch of MaskedSample instances with leading batch dim."""

    pitch_type_ids: torch.Tensor  # (batch, seq_len) long
    pitch_result_ids: torch.Tensor  # (batch, seq_len) long
    bb_type_ids: torch.Tensor  # (batch, seq_len) long
    stand_ids: torch.Tensor  # (batch, seq_len) long
    p_throws_ids: torch.Tensor  # (batch, seq_len) long
    pa_event_ids: torch.Tensor  # (batch, seq_len) long
    numeric_features: torch.Tensor  # (batch, seq_len, 23) float
    numeric_mask: torch.Tensor  # (batch, seq_len, 23) bool
    padding_mask: torch.Tensor  # (batch, seq_len) bool
    player_token_mask: torch.Tensor  # (batch, seq_len) bool
    game_ids: torch.Tensor  # (batch, seq_len) long
    target_pitch_type_ids: torch.Tensor  # (batch, seq_len) long
    target_pitch_result_ids: torch.Tensor  # (batch, seq_len) long
    mask_positions: torch.Tensor  # (batch, seq_len) bool
    seq_lengths: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


class MGMDataset(Dataset[MaskedSample]):
    """Dataset that applies BERT-style masking to tensorized pitch sequences.

    Call set_epoch() before each epoch to ensure different masking patterns.
    """

    def __init__(
        self,
        sequences: list[TensorizedSingle],
        config: PreTrainingConfig,
        pitch_type_vocab_size: int,
        pitch_result_vocab_size: int,
    ) -> None:
        self._sequences = sequences
        self._config = config
        self._pitch_type_vocab_size = pitch_type_vocab_size
        self._pitch_result_vocab_size = pitch_result_vocab_size
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch so masking patterns vary across epochs."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> MaskedSample:
        original = self._sequences[index]
        seq_len = original.seq_length

        # Clone all tensors to avoid mutating originals
        pitch_type_ids = original.pitch_type_ids.clone()
        pitch_result_ids = original.pitch_result_ids.clone()

        # Identify maskable positions: real pitch tokens only.
        # Exclude player tokens and the [CLS] token (game_id == -1).
        cls_mask = original.game_ids == -1
        maskable = original.padding_mask & ~original.player_token_mask & ~cls_mask

        # Deterministic RNG per (seed, epoch, idx) for reproducibility
        # with different masks each epoch
        rng = torch.Generator()
        rng.manual_seed(self._config.seed + self._epoch * len(self._sequences) + index)

        # Select ~mask_ratio fraction of maskable positions
        rand = torch.rand(seq_len, generator=rng)
        mask_positions = maskable & (rand < self._config.mask_ratio)

        # Build targets: original values at masked positions, 0 elsewhere
        target_pitch_type_ids = torch.zeros(seq_len, dtype=torch.long)
        target_pitch_result_ids = torch.zeros(seq_len, dtype=torch.long)
        target_pitch_type_ids[mask_positions] = original.pitch_type_ids[mask_positions]
        target_pitch_result_ids[mask_positions] = original.pitch_result_ids[mask_positions]

        # Apply BERT-style masking to inputs at masked positions
        n_masked = int(mask_positions.sum().item())
        if n_masked > 0:
            action_rand = torch.rand(n_masked, generator=rng)
            masked_indices = mask_positions.nonzero(as_tuple=True)[0]

            # 80% → zero out (replace with 0)
            zero_mask = action_rand < self._config.mask_replace_ratio
            # 10% → random replacement
            random_mask = (
                ~zero_mask
                & (action_rand < self._config.mask_replace_ratio + self._config.mask_random_ratio)
            )
            # 10% → keep original (no action needed)

            # Apply zeroing
            zero_indices = masked_indices[zero_mask]
            pitch_type_ids[zero_indices] = 0
            pitch_result_ids[zero_indices] = 0

            # Apply random replacement
            random_indices = masked_indices[random_mask]
            n_random = len(random_indices)
            if n_random > 0:
                # Random pitch type ids in [2, vocab_size) to avoid PAD=0 and UNK=1
                pitch_type_ids[random_indices] = torch.randint(
                    2, self._pitch_type_vocab_size, (n_random,), generator=rng
                )
                pitch_result_ids[random_indices] = torch.randint(
                    2, self._pitch_result_vocab_size, (n_random,), generator=rng
                )

        return MaskedSample(
            pitch_type_ids=pitch_type_ids,
            pitch_result_ids=pitch_result_ids,
            bb_type_ids=original.bb_type_ids.clone(),
            stand_ids=original.stand_ids.clone(),
            p_throws_ids=original.p_throws_ids.clone(),
            pa_event_ids=original.pa_event_ids.clone(),
            numeric_features=original.numeric_features.clone(),
            numeric_mask=original.numeric_mask.clone(),
            padding_mask=original.padding_mask.clone(),
            player_token_mask=original.player_token_mask.clone(),
            game_ids=original.game_ids.clone(),
            target_pitch_type_ids=target_pitch_type_ids,
            target_pitch_result_ids=target_pitch_result_ids,
            mask_positions=mask_positions,
            seq_length=seq_len,
        )


def collate_masked_samples(samples: list[MaskedSample]) -> MaskedBatch:
    """Pad and collate MaskedSamples into a MaskedBatch."""
    max_len = max(s.seq_length for s in samples)
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
    game_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_pitch_type_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_pitch_result_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask_positions = torch.zeros(batch_size, max_len, dtype=torch.bool)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, s in enumerate(samples):
        sl = s.seq_length
        pitch_type_ids[i, :sl] = s.pitch_type_ids
        pitch_result_ids[i, :sl] = s.pitch_result_ids
        bb_type_ids[i, :sl] = s.bb_type_ids
        stand_ids[i, :sl] = s.stand_ids
        p_throws_ids[i, :sl] = s.p_throws_ids
        pa_event_ids[i, :sl] = s.pa_event_ids
        numeric_features[i, :sl] = s.numeric_features
        numeric_mask[i, :sl] = s.numeric_mask
        padding_mask[i, :sl] = s.padding_mask
        player_token_mask[i, :sl] = s.player_token_mask
        game_ids[i, :sl] = s.game_ids
        target_pitch_type_ids[i, :sl] = s.target_pitch_type_ids
        target_pitch_result_ids[i, :sl] = s.target_pitch_result_ids
        mask_positions[i, :sl] = s.mask_positions
        seq_lengths[i] = sl

    return MaskedBatch(
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
        target_pitch_type_ids=target_pitch_type_ids,
        target_pitch_result_ids=target_pitch_result_ids,
        mask_positions=mask_positions,
        seq_lengths=seq_lengths,
    )


def masked_batch_to_tensorized_batch(batch: MaskedBatch) -> TensorizedBatch:
    """Extract input fields from MaskedBatch to create a TensorizedBatch for model.forward()."""
    return TensorizedBatch(
        pitch_type_ids=batch.pitch_type_ids,
        pitch_result_ids=batch.pitch_result_ids,
        bb_type_ids=batch.bb_type_ids,
        stand_ids=batch.stand_ids,
        p_throws_ids=batch.p_throws_ids,
        pa_event_ids=batch.pa_event_ids,
        numeric_features=batch.numeric_features,
        numeric_mask=batch.numeric_mask,
        padding_mask=batch.padding_mask,
        player_token_mask=batch.player_token_mask,
        game_ids=batch.game_ids,
        seq_lengths=batch.seq_lengths,
    )


def _build_season_pair(
    builder: GameSequenceBuilder,
    season: int,
    perspective: str,
) -> list[GameSequence]:
    """Build game sequences for a single (season, perspective) pair.

    Top-level function so it's picklable for multiprocessing.
    """
    return builder.build_season(season, perspective)


def build_player_contexts(
    builder: GameSequenceBuilder,
    seasons: tuple[int, ...],
    perspectives: tuple[str, ...],
    min_pitch_count: int,
    max_workers: int | None = None,
) -> list[PlayerContext]:
    """Build PlayerContext objects from Statcast data.

    Calls builder.build_season() for each (season, perspective) combination,
    groups by (player_id, season, perspective), creates PlayerContext objects,
    and filters by minimum pitch count.
    """
    from concurrent.futures import ProcessPoolExecutor

    from fantasy_baseball_manager.contextual.data.models import PlayerContext as PC

    # Collect all game sequences
    pairs = [(season, perspective) for season in seasons for perspective in perspectives]
    all_sequences: list[GameSequence] = []

    if len(pairs) > 1 and max_workers != 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_build_season_pair, builder, s, p) for s, p in pairs]
            for future in futures:
                all_sequences.extend(future.result())
    else:
        for season, perspective in pairs:
            all_sequences.extend(builder.build_season(season, perspective))

    # Group by (player_id, season, perspective)
    grouped: dict[tuple[int, int, str], list] = {}
    for seq in all_sequences:
        key = (seq.player_id, seq.season, seq.perspective)
        grouped.setdefault(key, []).append(seq)

    # Build PlayerContext objects and filter
    contexts: list[PlayerContext] = []
    for (player_id, season, perspective), games in grouped.items():
        # Sort by game_date for chronological order
        games.sort(key=lambda g: g.game_date)
        total_pitches = sum(len(g.pitches) for g in games)
        if total_pitches < min_pitch_count:
            continue
        ctx = PC(
            player_id=player_id,
            player_name="",  # Not available from Statcast data
            season=season,
            perspective=perspective,
            games=tuple(games),
        )
        contexts.append(ctx)

    return contexts


def _tensorize_single_context(
    ctx: PlayerContext,
    tensorizer: Tensorizer,
) -> TensorizedSingle:
    """Tensorize a single player context.

    Top-level function so it's picklable for multiprocessing.
    """
    return tensorizer.tensorize_context(ctx)


def tensorize_contexts(
    tensorizer: Tensorizer,
    contexts: list[PlayerContext],
    max_workers: int | None = None,
) -> list[TensorizedSingle]:
    """Tensorize player contexts, optionally in parallel.

    Uses ProcessPoolExecutor to parallelize tensorize_context calls
    when there are multiple contexts and max_workers != 1.
    """
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    if len(contexts) > 1 and max_workers != 1:
        fn = partial(_tensorize_single_context, tensorizer=tensorizer)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(fn, contexts, chunksize=16))
    else:
        return [tensorizer.tensorize_context(ctx) for ctx in contexts]


# ---------------------------------------------------------------------------
# Fine-tune stat extraction
# ---------------------------------------------------------------------------

# Mapping from target stat name to the set of pa_event values that count for it.
_STAT_EVENT_MAP: dict[str, set[str]] = {
    "hr": {"home_run"},
    "so": {"strikeout", "strikeout_double_play"},
    "bb": {"walk", "intentional_walk"},
    "h": {"single", "double", "triple", "home_run"},
    "2b": {"double"},
    "3b": {"triple"},
}


def extract_game_stats(
    game: GameSequence,
    target_stats: tuple[str, ...],
) -> torch.Tensor:
    """Extract counting stats from a game's pitches.

    Only counts pa_event on the last pitch of each plate appearance
    (where the next pitch has pitch_number == 1, or it's the final pitch)
    to avoid double-counting if pa_event is populated on every pitch in a PA.

    Returns:
        (n_targets,) float tensor of per-game counting stats.
    """
    counts = [0.0] * len(target_stats)
    pitches = game.pitches
    n_pitches = len(pitches)
    for idx, pitch in enumerate(pitches):
        if pitch.pa_event is None:
            continue
        # Only count on the last pitch of the PA
        is_last_in_pa = (
            idx == n_pitches - 1
            or pitches[idx + 1].pitch_number == 1
        )
        if not is_last_in_pa:
            continue
        for i, stat in enumerate(target_stats):
            events = _STAT_EVENT_MAP.get(stat)
            if events is not None and pitch.pa_event in events:
                counts[i] += 1.0
    return torch.tensor(counts, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Fine-tune dataset classes
# ---------------------------------------------------------------------------


@dataclass
class FineTuneSample:
    """A single fine-tuning training example."""

    context: TensorizedSingle
    targets: torch.Tensor  # (n_targets,) float
    context_mean: torch.Tensor  # (n_targets,) float — mean of context game stats


@dataclass
class FineTuneBatch:
    """Collated batch of FineTuneSample instances."""

    context: TensorizedBatch
    targets: torch.Tensor  # (batch, n_targets) float
    context_mean: torch.Tensor  # (batch, n_targets) float


def _build_player_windows(
    player_ctx: PlayerContext,
    tensorizer: Tensorizer,
    context_window: int,
    target_stats: tuple[str, ...],
) -> list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]]:
    """Build sliding windows for a single player.

    Top-level function so it's picklable for multiprocessing.

    Returns:
        List of (tensorized_context, targets, context_mean) tuples.
    """
    from fantasy_baseball_manager.contextual.data.models import PlayerContext as PC

    windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]] = []
    games = player_ctx.games
    n = context_window

    for i in range(len(games) - n):
        context_games = games[i : i + n]
        target_game = games[i + n]

        ctx = PC(
            player_id=player_ctx.player_id,
            player_name=player_ctx.player_name,
            season=player_ctx.season,
            perspective=player_ctx.perspective,
            games=context_games,
        )
        tensorized = tensorizer.tensorize_context(ctx)
        targets = extract_game_stats(target_game, target_stats)
        context_mean = torch.stack(
            [extract_game_stats(g, target_stats) for g in context_games]
        ).mean(dim=0)
        windows.append((tensorized, targets, context_mean))

    return windows


def build_finetune_windows(
    player_contexts: list[PlayerContext],
    tensorizer: Tensorizer,
    config: FineTuneConfig,
    target_stats: tuple[str, ...],
    max_workers: int | None = None,
) -> list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]]:
    """Build (context, targets, context_mean) triples using a sliding window.

    For each player with G games (where G >= context_window + 1),
    creates G - context_window examples:
      context = tensorized games[i:i+N], target = stats from games[i+N],
      context_mean = mean of stats over the N context games.
    """
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    n = config.context_window
    eligible = [p for p in player_contexts if len(p.games) >= n + 1]

    fn = partial(
        _build_player_windows,
        tensorizer=tensorizer,
        context_window=n,
        target_stats=target_stats,
    )

    windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]] = []

    if len(eligible) > 1 and max_workers != 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for player_windows in pool.map(fn, eligible, chunksize=8):
                windows.extend(player_windows)
    else:
        for ctx in eligible:
            windows.extend(fn(ctx))

    return windows


class FineTuneDataset(Dataset[FineTuneSample]):
    """Wraps pre-built sliding window examples for fine-tuning."""

    def __init__(self, windows: list[tuple[TensorizedSingle, torch.Tensor, torch.Tensor]]) -> None:
        self._windows = windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> FineTuneSample:
        context, targets, context_mean = self._windows[index]
        return FineTuneSample(context=context, targets=targets, context_mean=context_mean)


def collate_finetune_samples(samples: list[FineTuneSample]) -> FineTuneBatch:
    """Pad context fields and stack targets into a FineTuneBatch."""
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
    game_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
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

    return FineTuneBatch(context=context_batch, targets=targets, context_mean=context_mean)
