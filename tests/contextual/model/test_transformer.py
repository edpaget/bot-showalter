"""Tests for GamestateTransformer nn.Module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fantasy_baseball_manager.contextual.model.transformer import (
    GamestateTransformer,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig


class TestGamestateTransformer:
    """Tests for the GamestateTransformer module."""

    def test_output_shape(self, small_config: ModelConfig) -> None:
        transformer = GamestateTransformer(small_config)
        batch, seq_len = 2, 8
        embeddings = torch.randn(batch, seq_len, small_config.d_model)
        # No masking: all attend to all
        attn_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)

        out = transformer(embeddings, attn_mask, padding_mask)
        assert out.shape == (batch, seq_len, small_config.d_model)

    def test_mask_affects_output(self, small_config: ModelConfig) -> None:
        transformer = GamestateTransformer(small_config)
        transformer.eval()
        batch, seq_len = 1, 4
        embeddings = torch.randn(batch, seq_len, small_config.d_model)
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)

        # No mask
        no_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        out_no_mask = transformer(embeddings, no_mask, padding_mask)

        # Mask position 3 from all
        with_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        with_mask[0, :, 3] = True  # No one can attend to position 3
        out_with_mask = transformer(embeddings, with_mask, padding_mask)

        # Outputs should differ for positions that could see position 3
        assert not torch.allclose(out_no_mask[0, 0], out_with_mask[0, 0], atol=1e-5)

    def test_gradient_flow(self, small_config: ModelConfig) -> None:
        transformer = GamestateTransformer(small_config)
        batch, seq_len = 1, 4
        embeddings = torch.randn(batch, seq_len, small_config.d_model, requires_grad=True)
        attn_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)

        out = transformer(embeddings, attn_mask, padding_mask)
        loss = out.pow(2).sum()
        loss.backward()
        assert embeddings.grad is not None
        assert embeddings.grad.abs().sum() > 0

    def test_padding_mask_applied(self, small_config: ModelConfig) -> None:
        transformer = GamestateTransformer(small_config)
        transformer.eval()
        batch, seq_len = 1, 6
        embeddings = torch.randn(batch, seq_len, small_config.d_model)

        # All real
        attn_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        all_real = torch.ones(batch, seq_len, dtype=torch.bool)
        out_all = transformer(embeddings, attn_mask, all_real)

        # Last 2 positions padded
        partial = torch.ones(batch, seq_len, dtype=torch.bool)
        partial[0, 4:] = False
        out_partial = transformer(embeddings, attn_mask, partial)

        # Real positions should differ when padding changes
        assert not torch.allclose(out_all[0, 0], out_partial[0, 0], atol=1e-5)
