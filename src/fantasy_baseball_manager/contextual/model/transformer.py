"""Transformer encoder for gamestate sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig


class GamestateTransformer(nn.Module):
    """Transformer encoder over embedded pitch event sequences.

    Uses nn.TransformerEncoder with pre-norm (norm_first=True) for stable
    training. Accepts a 3D attention mask for player token isolation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Run the transformer encoder.

        Args:
            embeddings: (batch, seq_len, d_model)
            attention_mask: (batch, seq_len, seq_len) bool, True=MASKED
            padding_mask: (batch, seq_len) bool, True=real token

        Returns:
            (batch, seq_len, d_model) hidden states
        """
        # PyTorch TransformerEncoder expects:
        #   mask: (batch*n_heads, seq_len, seq_len) float with -inf for masked
        #   src_key_padding_mask: (batch, seq_len) bool, True=IGNORE
        n_heads = self._config.n_heads
        batch, seq_len, _ = embeddings.shape

        # Expand attention mask for multi-head: (batch, S, S) → (batch*n_heads, S, S)
        expanded_mask = (
            attention_mask.unsqueeze(1)
            .expand(batch, n_heads, seq_len, seq_len)
            .reshape(batch * n_heads, seq_len, seq_len)
        )

        # Convert bool mask to float mask (-inf for True/masked, 0 for False/attend)
        float_mask = torch.zeros_like(expanded_mask, dtype=embeddings.dtype)
        float_mask.masked_fill_(expanded_mask, float("-inf"))

        # Flip padding_mask: our True=real → PyTorch True=ignore
        # Use float mask to match the attention mask dtype and avoid deprecation warning
        key_padding_mask = torch.zeros_like(padding_mask, dtype=embeddings.dtype)
        key_padding_mask.masked_fill_(~padding_mask, float("-inf"))

        return self.encoder(
            embeddings,
            mask=float_mask,
            src_key_padding_mask=key_padding_mask,
        )
