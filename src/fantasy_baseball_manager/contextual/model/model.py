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
            player_emb = self._extract_player_embeddings(hidden, batch.player_token_mask)
            preds = self.head(player_emb)
            return {
                "performance_preds": preds,
                "transformer_output": hidden,
            }

    def swap_head(self, new_head: MaskedGamestateHead | PerformancePredictionHead) -> None:
        """Replace the current head while preserving embedder + transformer."""
        self.head = new_head

    def _extract_player_embeddings(self, hidden: Tensor, player_token_mask: Tensor) -> Tensor:
        """Gather hidden states at [PLAYER] token positions.

        Args:
            hidden: (batch, seq_len, d_model)
            player_token_mask: (batch, seq_len) bool

        Returns:
            (batch, n_player_tokens, d_model)
        """
        batch, _, d_model = hidden.shape
        n_players = int(player_token_mask[0].sum().item())

        result = torch.zeros(batch, n_players, d_model, dtype=hidden.dtype, device=hidden.device)
        for b in range(batch):
            indices = player_token_mask[b].nonzero(as_tuple=True)[0]
            result[b, : len(indices)] = hidden[b, indices]
        return result
