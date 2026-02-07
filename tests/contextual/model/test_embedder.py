"""Tests for EventEmbedder nn.Module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fantasy_baseball_manager.contextual.model.embedder import EventEmbedder

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig


class TestEventEmbedder:
    """Tests for the EventEmbedder module."""

    def test_output_shape(self, small_config: ModelConfig) -> None:
        embedder = EventEmbedder(small_config)
        batch, seq_len = 2, 5
        out = embedder(
            pitch_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pitch_result_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            bb_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            stand_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            p_throws_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pa_event_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            numeric_features=torch.randn(batch, seq_len, small_config.n_numeric_features),
            numeric_mask=torch.ones(batch, seq_len, small_config.n_numeric_features, dtype=torch.bool),
        )
        assert out.shape == (batch, seq_len, small_config.d_model)

    def test_pad_index_produces_zero_embeddings(self, small_config: ModelConfig) -> None:
        embedder = EventEmbedder(small_config)
        batch, seq_len = 1, 3
        # All categorical indices = 0 (PAD)
        out = embedder(
            pitch_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pitch_result_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            bb_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            stand_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            p_throws_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pa_event_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            numeric_features=torch.zeros(batch, seq_len, small_config.n_numeric_features),
            numeric_mask=torch.zeros(batch, seq_len, small_config.n_numeric_features, dtype=torch.bool),
        )
        # Output should still have shape but embedding inputs are zero for PAD
        assert out.shape == (batch, seq_len, small_config.d_model)

    def test_numeric_mask_zeros_preserved(self, small_config: ModelConfig) -> None:
        embedder = EventEmbedder(small_config)
        batch, seq_len = 1, 2
        numerics = torch.randn(batch, seq_len, small_config.n_numeric_features)
        mask_all_false = torch.zeros(batch, seq_len, small_config.n_numeric_features, dtype=torch.bool)

        out_masked = embedder(
            pitch_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pitch_result_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            bb_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            stand_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            p_throws_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pa_event_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            numeric_features=numerics,
            numeric_mask=mask_all_false,
        )

        # With all mask False, numerics should be zeroed before entering network
        out_zeros = embedder(
            pitch_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pitch_result_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            bb_type_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            stand_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            p_throws_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            pa_event_ids=torch.zeros(batch, seq_len, dtype=torch.long),
            numeric_features=torch.zeros_like(numerics),
            numeric_mask=mask_all_false,
        )
        assert torch.allclose(out_masked, out_zeros)

    def test_gradient_flow(self, small_config: ModelConfig) -> None:
        embedder = EventEmbedder(small_config)
        batch, seq_len = 1, 3
        numerics = torch.randn(batch, seq_len, small_config.n_numeric_features, requires_grad=True)

        out = embedder(
            pitch_type_ids=torch.ones(batch, seq_len, dtype=torch.long),
            pitch_result_ids=torch.ones(batch, seq_len, dtype=torch.long),
            bb_type_ids=torch.ones(batch, seq_len, dtype=torch.long),
            stand_ids=torch.ones(batch, seq_len, dtype=torch.long),
            p_throws_ids=torch.ones(batch, seq_len, dtype=torch.long),
            pa_event_ids=torch.ones(batch, seq_len, dtype=torch.long),
            numeric_features=numerics,
            numeric_mask=torch.ones(batch, seq_len, small_config.n_numeric_features, dtype=torch.bool),
        )
        # Use squared sum to avoid LayerNorm gradient cancellation with plain sum
        loss = out.pow(2).sum()
        loss.backward()
        assert numerics.grad is not None
        assert numerics.grad.abs().sum() > 0
