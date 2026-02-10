"""Tests for MGMDataset, masking logic, and collation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig
from fantasy_baseball_manager.contextual.training.dataset import (
    MaskedBatch,
    MaskedSample,
    MGMDataset,
    collate_masked_samples,
    compute_feature_statistics,
    masked_batch_to_tensorized_batch,
)
from tests.contextual.model.conftest import make_player_context


def _build_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


def _make_sequences(config: ModelConfig, n_games: int = 2, pitches_per_game: int = 10) -> list:

    tensorizer = _build_tensorizer(config)
    ctx = make_player_context(n_games=n_games, pitches_per_game=pitches_per_game)
    return [tensorizer.tensorize_context(ctx)]


class TestMGMDataset:
    def test_len(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config)
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        assert len(dataset) == 1

    def test_getitem_returns_masked_sample(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config)
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        assert isinstance(sample, MaskedSample)
        seq_len = sequences[0].seq_length
        assert sample.pitch_type_ids.shape == (seq_len,)
        assert sample.pitch_result_ids.shape == (seq_len,)
        assert sample.mask_positions.shape == (seq_len,)
        assert sample.target_pitch_type_ids.shape == (seq_len,)
        assert sample.target_pitch_result_ids.shape == (seq_len,)

    def test_mask_ratio_approximate(self, small_config: ModelConfig) -> None:
        """Approximately 15% of maskable positions should be selected."""
        sequences = _make_sequences(small_config, n_games=3, pitches_per_game=20)
        config = PreTrainingConfig(mask_ratio=0.15, seed=42)
        dataset = MGMDataset(
            sequences=sequences,
            config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        # Count maskable positions (real pitch tokens, excluding player tokens and CLS)
        cls_mask = sequences[0].game_ids == -1
        maskable = sequences[0].padding_mask & ~sequences[0].player_token_mask & ~cls_mask
        n_maskable = int(maskable.sum().item())
        n_masked = int(sample.mask_positions.sum().item())
        # Allow tolerance
        expected = int(n_maskable * 0.15)
        assert abs(n_masked - expected) <= max(3, int(n_maskable * 0.05))

    def test_player_tokens_never_masked(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config, n_games=2, pitches_per_game=15)
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(mask_ratio=0.5, seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        player_mask = sequences[0].player_token_mask
        # No player tokens should be masked
        assert not (sample.mask_positions & player_mask).any()

    def test_cls_token_never_masked(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config, n_games=2, pitches_per_game=15)
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(mask_ratio=0.5, seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        cls_mask = sequences[0].game_ids == -1
        assert not (sample.mask_positions & cls_mask).any()

    def test_padding_tokens_never_masked(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config, n_games=1, pitches_per_game=5)
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(mask_ratio=0.5, seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        padding = ~sequences[0].padding_mask
        assert not (sample.mask_positions & padding).any()

    def test_targets_contain_original_at_masked_positions(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config, n_games=2, pitches_per_game=10)
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(mask_ratio=0.3, seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        original = sequences[0]
        # At masked positions, targets should match originals
        masked = sample.mask_positions
        assert torch.equal(
            sample.target_pitch_type_ids[masked],
            original.pitch_type_ids[masked],
        )
        assert torch.equal(
            sample.target_pitch_result_ids[masked],
            original.pitch_result_ids[masked],
        )

    def test_targets_sentinel_at_unmasked_positions(self, small_config: ModelConfig) -> None:
        """Non-masked positions use -100 sentinel (PyTorch ignore_index convention)."""
        sequences = _make_sequences(small_config, n_games=2, pitches_per_game=10)
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(mask_ratio=0.3, seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        unmasked = ~sample.mask_positions
        assert (sample.target_pitch_type_ids[unmasked] == -100).all()
        assert (sample.target_pitch_result_ids[unmasked] == -100).all()

    def test_masked_pad_index_zero_not_ignored(self, small_config: ModelConfig) -> None:
        """If a masked position has true label 0 (PAD index), it appears in the target."""
        sequences = _make_sequences(small_config, n_games=2, pitches_per_game=10)
        original = sequences[0]
        # Force a maskable pitch position to have pitch_type_id=0 (PAD index)
        cls_mask = original.game_ids == -1
        maskable = original.padding_mask & ~original.player_token_mask & ~cls_mask
        maskable_indices = maskable.nonzero(as_tuple=True)[0]
        assert len(maskable_indices) > 0
        forced_idx = maskable_indices[0].item()
        original.pitch_type_ids[forced_idx] = 0

        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(mask_ratio=1.0, seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        # The forced position should be masked and its target should be 0 (not -100)
        assert sample.mask_positions[forced_idx].item() is True
        assert sample.target_pitch_type_ids[forced_idx].item() == 0

    def test_bert_masking_80_percent_zeroed(self, small_config: ModelConfig) -> None:
        """80% of masked tokens should have categorical ids zeroed out."""
        # Use a high mask ratio for more masked tokens to test distribution
        sequences = _make_sequences(small_config, n_games=5, pitches_per_game=30)
        config = PreTrainingConfig(
            mask_ratio=0.5, mask_replace_ratio=0.8, mask_random_ratio=0.1, seed=42
        )
        dataset = MGMDataset(
            sequences=sequences,
            config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        sample = dataset[0]
        masked = sample.mask_positions
        n_masked = int(masked.sum().item())
        # Of masked positions, count how many have pitch_type_ids == 0 (zeroed)
        zeroed = (sample.pitch_type_ids[masked] == 0).sum().item()
        # Should be roughly 80% ± tolerance
        ratio = zeroed / n_masked if n_masked > 0 else 0
        assert 0.6 < ratio < 1.0

    def test_pa_event_zeroed_at_masked_positions(self, small_config: ModelConfig) -> None:
        """pa_event_ids should be zeroed where pitch_type_ids is zeroed."""
        sequences = _make_sequences(small_config, n_games=5, pitches_per_game=30)
        config = PreTrainingConfig(
            mask_ratio=0.5, mask_replace_ratio=1.0, mask_random_ratio=0.0, seed=42,
        )
        dataset = MGMDataset(
            sequences=sequences,
            config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
            pa_event_vocab_size=PA_EVENT_VOCAB.size,
        )
        sample = dataset[0]
        masked = sample.mask_positions
        # With 100% replace ratio, all masked positions should have pa_event_ids == 0
        assert (sample.pa_event_ids[masked] == 0).all()

    def test_pa_event_randomized_at_random_positions(self, small_config: ModelConfig) -> None:
        """pa_event_ids should be randomized where pitch_type_ids is randomized."""
        sequences = _make_sequences(small_config, n_games=5, pitches_per_game=30)
        config = PreTrainingConfig(
            mask_ratio=0.5, mask_replace_ratio=0.0, mask_random_ratio=1.0, seed=42,
        )
        dataset = MGMDataset(
            sequences=sequences,
            config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
            pa_event_vocab_size=PA_EVENT_VOCAB.size,
        )
        sample = dataset[0]
        original = sequences[0]
        masked = sample.mask_positions
        n_masked = int(masked.sum().item())
        if n_masked > 0:
            # With 100% random ratio, pa_event_ids at masked positions should differ
            # from originals (statistically — at least some should differ)
            changed = (sample.pa_event_ids[masked] != original.pa_event_ids[masked]).sum().item()
            assert changed > 0, "pa_event_ids should be randomized at random-replace positions"

    def test_pa_event_unchanged_at_keep_positions(self, small_config: ModelConfig) -> None:
        """pa_event_ids should be unchanged at keep (10%) and unmasked positions."""
        sequences = _make_sequences(small_config, n_games=5, pitches_per_game=30)
        config = PreTrainingConfig(
            mask_ratio=0.5, mask_replace_ratio=0.0, mask_random_ratio=0.0, seed=42,
        )
        dataset = MGMDataset(
            sequences=sequences,
            config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
            pa_event_vocab_size=PA_EVENT_VOCAB.size,
        )
        sample = dataset[0]
        original = sequences[0]
        # With 0% replace and 0% random → 100% keep, all masked positions keep originals
        assert torch.equal(sample.pa_event_ids, original.pa_event_ids)

    def test_does_not_mutate_original(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config)
        original_pt = sequences[0].pitch_type_ids.clone()
        dataset = MGMDataset(
            sequences=sequences,
            config=PreTrainingConfig(seed=42),
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        _ = dataset[0]
        assert torch.equal(sequences[0].pitch_type_ids, original_pt)

    def test_deterministic_with_same_seed(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config, n_games=2, pitches_per_game=10)
        config = PreTrainingConfig(seed=42)
        ds1 = MGMDataset(
            sequences=sequences, config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        ds2 = MGMDataset(
            sequences=sequences, config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        s1 = ds1[0]
        s2 = ds2[0]
        assert torch.equal(s1.mask_positions, s2.mask_positions)
        assert torch.equal(s1.pitch_type_ids, s2.pitch_type_ids)

    def test_masking_varies_across_epochs(self, small_config: ModelConfig) -> None:
        """set_epoch() should produce different masking patterns."""
        sequences = _make_sequences(small_config, n_games=3, pitches_per_game=20)
        config = PreTrainingConfig(mask_ratio=0.3, seed=42)
        dataset = MGMDataset(
            sequences=sequences, config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )

        dataset.set_epoch(0)
        s0 = dataset[0]

        dataset.set_epoch(1)
        s1 = dataset[0]

        # Masks should differ between epochs
        assert not torch.equal(s0.mask_positions, s1.mask_positions)


class TestCollation:
    def test_collate_shapes(self, small_config: ModelConfig) -> None:
        tensorizer = _build_tensorizer(small_config)
        ctx1 = make_player_context(n_games=1, pitches_per_game=5)
        ctx2 = make_player_context(n_games=2, pitches_per_game=8)
        seqs = [tensorizer.tensorize_context(ctx1), tensorizer.tensorize_context(ctx2)]
        config = PreTrainingConfig(seed=42)
        dataset = MGMDataset(
            sequences=seqs, config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        samples = [dataset[0], dataset[1]]
        batch = collate_masked_samples(samples)
        assert isinstance(batch, MaskedBatch)
        max_len = max(s.seq_length for s in seqs)
        assert batch.pitch_type_ids.shape == (2, max_len)
        assert batch.mask_positions.shape == (2, max_len)
        assert batch.target_pitch_type_ids.shape == (2, max_len)
        assert batch.target_pitch_result_ids.shape == (2, max_len)
        assert batch.padding_mask.shape == (2, max_len)

    def test_masked_batch_to_tensorized_batch(self, small_config: ModelConfig) -> None:
        sequences = _make_sequences(small_config)
        config = PreTrainingConfig(seed=42)
        dataset = MGMDataset(
            sequences=sequences, config=config,
            pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
            pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        )
        batch = collate_masked_samples([dataset[0]])
        tb = masked_batch_to_tensorized_batch(batch)
        assert tb.pitch_type_ids.shape == batch.pitch_type_ids.shape
        assert tb.padding_mask.shape == batch.padding_mask.shape
        assert torch.equal(tb.pitch_type_ids, batch.pitch_type_ids)


class TestComputeFeatureStatistics:
    def test_known_values(self) -> None:
        """Mean and std match hand-computed values for known inputs."""
        from fantasy_baseball_manager.contextual.model.tensorizer import (
            NUMERIC_FIELDS,
            TensorizedSingle,
        )

        n_features = len(NUMERIC_FIELDS)
        seq_len = 4

        # All features present, values [1, 2, 3, 4] for feature 0, all zeros elsewhere
        numeric = torch.zeros(seq_len, n_features)
        numeric[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.zeros(seq_len, n_features, dtype=torch.bool)
        mask[:, 0] = True

        seq = TensorizedSingle(
            pitch_type_ids=torch.zeros(seq_len, dtype=torch.long),
            pitch_result_ids=torch.zeros(seq_len, dtype=torch.long),
            bb_type_ids=torch.zeros(seq_len, dtype=torch.long),
            stand_ids=torch.zeros(seq_len, dtype=torch.long),
            p_throws_ids=torch.zeros(seq_len, dtype=torch.long),
            pa_event_ids=torch.zeros(seq_len, dtype=torch.long),
            numeric_features=numeric,
            numeric_mask=mask,
            padding_mask=torch.ones(seq_len, dtype=torch.bool),
            player_token_mask=torch.zeros(seq_len, dtype=torch.bool),
            game_ids=torch.zeros(seq_len, dtype=torch.long),
            seq_length=seq_len,
        )

        mean, std = compute_feature_statistics([seq])
        assert mean.shape == (n_features,)
        assert std.shape == (n_features,)
        # Feature 0: mean=2.5, std=sqrt(1.25)≈1.118
        assert abs(mean[0].item() - 2.5) < 1e-5
        expected_std = (torch.tensor([1.0, 2.0, 3.0, 4.0]).var(correction=0)).sqrt()
        assert abs(std[0].item() - expected_std.item()) < 1e-4
        # Unmasked features should have std clamped to >= 1e-6 (not zero)
        assert (std >= 1e-6).all()

    def test_multiple_sequences(self) -> None:
        """Statistics accumulate correctly across multiple sequences."""
        from fantasy_baseball_manager.contextual.model.tensorizer import (
            NUMERIC_FIELDS,
            TensorizedSingle,
        )

        n_features = len(NUMERIC_FIELDS)

        def _make_seq(values: list[float]) -> TensorizedSingle:
            seq_len = len(values)
            numeric = torch.zeros(seq_len, n_features)
            numeric[:, 0] = torch.tensor(values)
            mask = torch.zeros(seq_len, n_features, dtype=torch.bool)
            mask[:, 0] = True
            return TensorizedSingle(
                pitch_type_ids=torch.zeros(seq_len, dtype=torch.long),
                pitch_result_ids=torch.zeros(seq_len, dtype=torch.long),
                bb_type_ids=torch.zeros(seq_len, dtype=torch.long),
                stand_ids=torch.zeros(seq_len, dtype=torch.long),
                p_throws_ids=torch.zeros(seq_len, dtype=torch.long),
                pa_event_ids=torch.zeros(seq_len, dtype=torch.long),
                numeric_features=numeric,
                numeric_mask=mask,
                padding_mask=torch.ones(seq_len, dtype=torch.bool),
                player_token_mask=torch.zeros(seq_len, dtype=torch.bool),
                game_ids=torch.zeros(seq_len, dtype=torch.long),
                seq_length=seq_len,
            )

        seq1 = _make_seq([10.0, 20.0])
        seq2 = _make_seq([30.0, 40.0])
        mean, _std = compute_feature_statistics([seq1, seq2])
        # Combined: [10, 20, 30, 40] → mean=25
        assert abs(mean[0].item() - 25.0) < 1e-4
