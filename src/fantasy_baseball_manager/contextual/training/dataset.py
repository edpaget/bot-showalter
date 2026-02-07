"""MGM dataset with BERT-style masking for pre-training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from fantasy_baseball_manager.contextual.model.tensorizer import (
    NUMERIC_FIELDS,
    TensorizedBatch,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.models import PlayerContext
    from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedSingle
    from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig


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
    """Dataset that applies BERT-style masking to tensorized pitch sequences."""

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

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> MaskedSample:
        original = self._sequences[index]
        seq_len = original.seq_length

        # Clone all tensors to avoid mutating originals
        pitch_type_ids = original.pitch_type_ids.clone()
        pitch_result_ids = original.pitch_result_ids.clone()

        # Identify maskable positions: real tokens that are not player tokens
        maskable = original.padding_mask & ~original.player_token_mask

        # Deterministic RNG per (seed, idx) for reproducibility
        rng = torch.Generator()
        rng.manual_seed(self._config.seed + index)

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


def build_player_contexts(
    builder: GameSequenceBuilder,
    seasons: tuple[int, ...],
    perspectives: tuple[str, ...],
    min_pitch_count: int,
) -> list[PlayerContext]:
    """Build PlayerContext objects from Statcast data.

    Calls builder.build_season() for each (season, perspective) combination,
    groups by (player_id, season, perspective), creates PlayerContext objects,
    and filters by minimum pitch count.
    """
    from fantasy_baseball_manager.contextual.data.models import PlayerContext as PC

    # Collect all game sequences
    all_sequences = []
    for season in seasons:
        for perspective in perspectives:
            sequences = builder.build_season(season, perspective)
            all_sequences.extend(sequences)

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
