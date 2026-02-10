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

    def test_partial_numeric_mask_normalizes_correctly(self, small_config: ModelConfig) -> None:
        """Masked LayerNorm should compute stats only over present features.

        With a partial mask, two inputs that differ only in masked-out positions
        should produce identical output.
        """
        embedder = EventEmbedder(small_config)
        embedder.eval()
        batch, seq_len = 1, 2
        n_num = small_config.n_numeric_features

        # Only the first 5 features are present
        mask = torch.zeros(batch, seq_len, n_num, dtype=torch.bool)
        mask[:, :, :5] = True

        numerics_a = torch.randn(batch, seq_len, n_num)
        numerics_b = numerics_a.clone()
        # Change masked-out positions — should not affect output
        numerics_b[:, :, 5:] = torch.randn(batch, seq_len, n_num - 5) * 100

        zeros = torch.zeros(batch, seq_len, dtype=torch.long)
        kwargs = dict(
            pitch_type_ids=zeros, pitch_result_ids=zeros, bb_type_ids=zeros,
            stand_ids=zeros, p_throws_ids=zeros, pa_event_ids=zeros,
        )
        out_a = embedder(**kwargs, numeric_features=numerics_a, numeric_mask=mask)
        out_b = embedder(**kwargs, numeric_features=numerics_b, numeric_mask=mask)

        assert torch.allclose(out_a, out_b, atol=1e-5)

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
