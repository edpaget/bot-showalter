"""Top-level contextual performance model composing all components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from fantasy_baseball_manager.contextual.model.embedder import EventEmbedder
from fantasy_baseball_manager.contextual.model.heads import (
    MaskedGamestateHead,
    PerformancePredictionHead,
)
from fantasy_baseball_manager.contextual.model.mask import build_player_attention_mask
from fantasy_baseball_manager.contextual.model.positional import (
    SinusoidalPositionalEncoding,
)
from fantasy_baseball_manager.contextual.model.transformer import (
    GamestateTransformer,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedBatch


class ContextualPerformanceModel(nn.Module):
    """Transformer model for contextual player performance prediction.

    Composes: EventEmbedder → SinusoidalPositionalEncoding → GamestateTransformer → Head.

    The head can be either:
      - MaskedGamestateHead for pre-training (predicts masked pitch type/result)
      - PerformancePredictionHead for fine-tuning (predicts performance stats)

    Use swap_head() to switch between pre-training and fine-tuning while
    preserving the embedder and transformer weights.
    """

    def __init__(
        self,
        config: ModelConfig,
        head: MaskedGamestateHead | PerformancePredictionHead,
    ) -> None:
        super().__init__()
        self.embedder = EventEmbedder(config)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.transformer = GamestateTransformer(config)
        self.head = head
        self.cls_embedding = nn.Parameter(torch.randn(config.d_model) * 0.02)

    def forward(self, batch: TensorizedBatch) -> dict[str, Tensor]:
        """Run the full model pipeline.

        Args:
            batch: TensorizedBatch with all tensor fields.

        Returns:
            Dict with task-specific outputs:
              MGM mode: {"pitch_type_logits", "pitch_result_logits", "transformer_output"}
              Perf mode: {"performance_preds", "transformer_output"}
        """
        # 1. Embed
        embeddings = self.embedder(
            pitch_type_ids=batch.pitch_type_ids,
            pitch_result_ids=batch.pitch_result_ids,
            bb_type_ids=batch.bb_type_ids,
            stand_ids=batch.stand_ids,
            p_throws_ids=batch.p_throws_ids,
            pa_event_ids=batch.pa_event_ids,
            numeric_features=batch.numeric_features,
            numeric_mask=batch.numeric_mask,
        )

        # 1b. Replace CLS positions with learned embedding
        cls_positions = batch.game_ids == -1  # (batch, seq_len)
        embeddings[cls_positions] = self.cls_embedding.to(embeddings.dtype)

        # 2. Add positional encoding
        embeddings = self.positional_encoding(embeddings)

        # 3. Build attention mask
        attn_mask = build_player_attention_mask(
            padding_mask=batch.padding_mask,
            player_token_mask=batch.player_token_mask,
            game_ids=batch.game_ids,
        )

        # 4. Transformer
        hidden = self.transformer(embeddings, attn_mask, batch.padding_mask)

        # 5. Head
        if isinstance(self.head, MaskedGamestateHead):
            pitch_type_logits, pitch_result_logits = self.head(hidden)
            return {
                "pitch_type_logits": pitch_type_logits,
                "pitch_result_logits": pitch_result_logits,
                "transformer_output": hidden,
            }
        else:
            cls_emb = self._extract_cls_embedding(hidden)
            preds = self.head(cls_emb)
            return {
                "performance_preds": preds,  # (batch, n_targets)
                "transformer_output": hidden,
            }

    def swap_head(self, new_head: MaskedGamestateHead | PerformancePredictionHead) -> None:
        """Replace the current head while preserving embedder + transformer."""
        self.head = new_head

    def _extract_cls_embedding(self, hidden: Tensor) -> Tensor:
        """Extract the [CLS] token hidden state at position 0.

        Args:
            hidden: (batch, seq_len, d_model)

        Returns:
            (batch, d_model)
        """
        return hidden[:, 0, :]
