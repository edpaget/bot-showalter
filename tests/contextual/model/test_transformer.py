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
        """Padding encoded in 3D attention mask affects real-token output."""
        transformer = GamestateTransformer(small_config)
        transformer.eval()
        batch, seq_len = 1, 6
        embeddings = torch.randn(batch, seq_len, small_config.d_model)
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)

        # All real — no masking
        no_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        out_all = transformer(embeddings, no_mask, padding_mask)

        # Last 2 positions padded — block padding columns in 3D mask
        with_padding = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        with_padding[0, :, 4:] = True  # no one attends to padding
        with_padding[0, 4:, :4] = True  # padding doesn't attend to real
        out_partial = transformer(embeddings, with_padding, padding_mask)

        # Real positions should differ when padding changes
        assert not torch.allclose(out_all[0, 0], out_partial[0, 0], atol=1e-5)

    def test_padding_positions_no_nan(self, small_config: ModelConfig) -> None:
        """Padding positions must produce finite output, not NaN.

        Without diagonal self-attention, padding rows get all -inf in the
        attention mask, causing softmax to output NaN in the math SDPA backend.
        """
        transformer = GamestateTransformer(small_config)
        transformer.eval()
        batch, seq_len = 2, 8

        embeddings = torch.randn(batch, seq_len, small_config.d_model)

        # Mask: padding positions (last 3) can't attend to anything except self
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        padding_mask[:, 5:] = False

        # Build a mask where padding columns are blocked for all queries
        attn_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        # Block padding columns (no one attends to padding)
        attn_mask[:, :, 5:] = True
        # Block padding rows from attending to real tokens
        attn_mask[:, 5:, :5] = True

        out = transformer(embeddings, attn_mask, padding_mask)

        # No NaN anywhere in output
        assert not torch.isnan(out).any(), "Transformer output contains NaN values"
        assert torch.isfinite(out).all(), "Transformer output contains non-finite values"

    def test_padding_no_nan_multi_layer(self) -> None:
        """NaN from padding must not propagate through multiple transformer layers.

        With n_layers=1 padding NaN stays isolated, but with n_layers>=2 NaN
        hidden states at padding positions produce NaN K/V in layer 2, which
        corrupts real positions when Q_real @ K_NaN = NaN in the attention logits.
        """
        from fantasy_baseball_manager.contextual.model.config import ModelConfig

        config = ModelConfig(d_model=32, n_layers=3, n_heads=2, ff_dim=64, dropout=0.0)
        transformer = GamestateTransformer(config)
        transformer.eval()
        batch, seq_len = 2, 10

        embeddings = torch.randn(batch, seq_len, config.d_model)

        # Padding: last 4 positions are padding
        padding_mask = torch.ones(batch, seq_len, dtype=torch.bool)
        padding_mask[:, 6:] = False

        attn_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
        attn_mask[:, :, 6:] = True   # no one attends to padding
        attn_mask[:, 6:, :6] = True   # padding doesn't attend to real

        out = transformer(embeddings, attn_mask, padding_mask)

        # Real positions must be NaN-free
        assert not torch.isnan(out[:, :6]).any(), (
            "Real positions contain NaN — padding NaN propagated through layers"
        )
        assert torch.isfinite(out).all(), "Output contains non-finite values"
