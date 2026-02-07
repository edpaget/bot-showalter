"""Task-specific prediction heads for the contextual model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig


class MaskedGamestateHead(nn.Module):
    """Pre-training head that predicts masked pitch type and result.

    Architecture: Linear → GELU → LayerNorm → two parallel Linear heads.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.shared = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.pitch_type_head = nn.Linear(d, config.pitch_type_vocab_size)
        self.pitch_result_head = nn.Linear(d, config.pitch_result_vocab_size)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Predict pitch type and result logits.

        Args:
            hidden_states: (..., d_model)

        Returns:
            (pitch_type_logits: ..., vocab_size), (pitch_result_logits: ..., vocab_size)
        """
        shared = self.shared(hidden_states)
        return self.pitch_type_head(shared), self.pitch_result_head(shared)


class PerformancePredictionHead(nn.Module):
    """Fine-tuning head that predicts player performance stats.

    Architecture: Linear → GELU → LayerNorm → Dropout → Linear.
    Parameterized by n_targets so the same class works for batters and pitchers.
    """

    def __init__(self, config: ModelConfig, n_targets: int) -> None:
        super().__init__()
        d = config.d_model
        self.network = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
            nn.Linear(d, n_targets),
        )

    def forward(self, player_embeddings: Tensor) -> Tensor:
        """Predict performance stats from player embeddings.

        Args:
            player_embeddings: (..., d_model)

        Returns:
            (..., n_targets)
        """
        return self.network(player_embeddings)
