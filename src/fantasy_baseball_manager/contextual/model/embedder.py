"""Event embedder that maps tensorized pitch features to d_model vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig


class EventEmbedder(nn.Module):
    """Maps tensorized pitch features to d_model-dimensional vectors.

    Architecture:
      1. Six nn.Embedding tables (one per categorical, padding_idx=0)
      2. Numeric branch: zero masked values → LayerNorm → Linear
      3. Concatenate all → Linear(cat_dim, d_model) → LayerNorm → Dropout
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._config = config

        # Categorical embeddings
        self.pitch_type_emb = nn.Embedding(config.pitch_type_vocab_size, config.pitch_type_embed_dim, padding_idx=0)
        self.pitch_result_emb = nn.Embedding(
            config.pitch_result_vocab_size, config.pitch_result_embed_dim, padding_idx=0
        )
        self.bb_type_emb = nn.Embedding(config.bb_type_vocab_size, config.bb_type_embed_dim, padding_idx=0)
        self.stand_emb = nn.Embedding(config.handedness_vocab_size, config.stand_embed_dim, padding_idx=0)
        self.p_throws_emb = nn.Embedding(config.handedness_vocab_size, config.p_throws_embed_dim, padding_idx=0)
        self.pa_event_emb = nn.Embedding(config.pa_event_vocab_size, config.pa_event_embed_dim, padding_idx=0)

        # Numeric branch
        n_num = config.n_numeric_features
        self.numeric_norm = nn.LayerNorm(n_num)
        self.numeric_proj = nn.Linear(n_num, n_num)

        # Projection to d_model
        cat_dim = (
            config.pitch_type_embed_dim
            + config.pitch_result_embed_dim
            + config.bb_type_embed_dim
            + config.stand_embed_dim
            + config.p_throws_embed_dim
            + config.pa_event_embed_dim
            + n_num
        )
        self.projection = nn.Linear(cat_dim, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        pitch_type_ids: Tensor,
        pitch_result_ids: Tensor,
        bb_type_ids: Tensor,
        stand_ids: Tensor,
        p_throws_ids: Tensor,
        pa_event_ids: Tensor,
        numeric_features: Tensor,
        numeric_mask: Tensor,
    ) -> Tensor:
        """Embed all pitch features into d_model vectors.

        Args:
            pitch_type_ids: (batch, seq_len) long
            pitch_result_ids: (batch, seq_len) long
            bb_type_ids: (batch, seq_len) long
            stand_ids: (batch, seq_len) long
            p_throws_ids: (batch, seq_len) long
            pa_event_ids: (batch, seq_len) long
            numeric_features: (batch, seq_len, n_numeric) float
            numeric_mask: (batch, seq_len, n_numeric) bool

        Returns:
            (batch, seq_len, d_model) float tensor
        """
        # Categorical embeddings
        pt = self.pitch_type_emb(pitch_type_ids)
        pr = self.pitch_result_emb(pitch_result_ids)
        bb = self.bb_type_emb(bb_type_ids)
        st = self.stand_emb(stand_ids)
        pt_ = self.p_throws_emb(p_throws_ids)
        pa = self.pa_event_emb(pa_event_ids)

        # Numeric branch: mask → norm → re-mask → linear
        masked_numeric = numeric_features * numeric_mask.float()
        normed = self.numeric_norm(masked_numeric)
        normed = normed * numeric_mask.float()  # Re-mask after norm
        numeric_out = self.numeric_proj(normed)

        # Concatenate and project
        combined = torch.cat([pt, pr, bb, st, pt_, pa, numeric_out], dim=-1)
        projected = self.projection(combined)
        return self.dropout(self.layer_norm(projected))
