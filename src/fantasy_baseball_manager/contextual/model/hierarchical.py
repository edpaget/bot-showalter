"""Hierarchical model with identity conditioning (Phase 2a).

Architecture:
  Identity Branch: stat MLP + archetype embedding → identity_repr
  Pitch Branch (frozen backbone): EventEmbedder → PosEnc → Transformer → hidden states
  Game Pooler: mean-pool hidden states per game
  Level 3 Attention: [PROJ] token + identity bias + game sequence → transformer
  Prediction Head: cat(proj_output, identity_repr) → targets
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from fantasy_baseball_manager.contextual.model.tensorizer import TensorizedBatch

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.hierarchical_config import (
        HierarchicalModelConfig,
    )
    from fantasy_baseball_manager.contextual.model.model import (
        ContextualPerformanceModel,
    )


class IdentityModule(nn.Module):
    """Produces identity representation from stat features and archetype ID.

    Architecture:
      stat_features → Linear(input→128) → GELU → Linear(128→stat_dim) → LayerNorm → stat_repr
      archetype_id → Embedding(n_archetypes, archetype_dim) → arch_repr
      identity_repr = cat(stat_repr, arch_repr)
    """

    def __init__(self, config: HierarchicalModelConfig, stat_input_dim: int) -> None:
        super().__init__()
        self.stat_mlp = nn.Sequential(
            nn.Linear(stat_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, config.identity_stat_dim),
            nn.LayerNorm(config.identity_stat_dim),
        )
        self.archetype_emb = nn.Embedding(config.n_archetypes, config.identity_archetype_dim)

    def forward(self, stat_features: Tensor, archetype_ids: Tensor) -> Tensor:
        """Compute identity representation.

        Args:
            stat_features: (batch, stat_input_dim)
            archetype_ids: (batch,) long

        Returns:
            (batch, identity_repr_dim)
        """
        stat_repr = self.stat_mlp(stat_features)
        arch_repr = self.archetype_emb(archetype_ids)
        return torch.cat([stat_repr, arch_repr], dim=-1)


class GamePooler(nn.Module):
    """Mean-pools pitch-level hidden states per game.

    Excludes CLS tokens (game_id == -1), PLAYER tokens, and padding.
    """

    def forward(
        self,
        hidden: Tensor,
        game_ids: Tensor,
        padding_mask: Tensor,
        player_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pool hidden states per game.

        Args:
            hidden: (batch, seq_len, d_model)
            game_ids: (batch, seq_len) — non-negative for pitch tokens, -1 for CLS, -2 for pad
            padding_mask: (batch, seq_len) bool — True=real token
            player_token_mask: (batch, seq_len) bool — True=player token

        Returns:
            game_embeddings: (batch, max_n_games, d_model)
            game_mask: (batch, max_n_games) bool — True=valid game
        """
        batch_size, _seq_len, d_model = hidden.shape

        # Only consider real pitch tokens (not CLS, not player, not padding)
        pitch_mask = padding_mask & ~player_token_mask & (game_ids >= 0)

        # Find max game count across batch
        n_games = 1 if not pitch_mask.any() else int(game_ids[pitch_mask].max().item()) + 1

        # Remap non-pitch game_ids to 0 so scatter indices are non-negative.
        # Their contributions are zeroed by pitch_mask so they don't pollute.
        safe_ids = game_ids.clamp(min=0)  # (batch, seq_len)

        # Scatter-add hidden states per game
        masked_hidden = hidden * pitch_mask.unsqueeze(-1)  # zero non-pitch
        idx = safe_ids.unsqueeze(-1).expand(-1, -1, d_model)  # (batch, seq_len, d_model)
        game_sums = hidden.new_zeros(batch_size, n_games, d_model)
        game_sums.scatter_add_(1, idx, masked_hidden)

        # Count pitch tokens per game
        game_counts = hidden.new_zeros(batch_size, n_games)
        game_counts.scatter_add_(1, safe_ids, pitch_mask.to(dtype=hidden.dtype))

        # Mean pool (clamp avoids division by zero for empty games)
        game_mask = game_counts > 0
        game_embeddings = game_sums / game_counts.unsqueeze(-1).clamp(min=1)

        return game_embeddings, game_mask


class Level3Attention(nn.Module):
    """Game-level transformer with identity-conditioned [PROJ] token.

    Prepends a learnable [PROJ] token biased by projected identity to
    the game embedding sequence, runs a small transformer, and extracts
    the [PROJ] output.
    """

    def __init__(self, config: HierarchicalModelConfig) -> None:
        super().__init__()
        d = config.level3_d_model
        self.identity_proj = nn.Linear(config.identity_repr_dim, d)
        self.proj_token = nn.Parameter(torch.randn(d) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.level3_n_heads,
            dim_feedforward=config.level3_ff_dim,
            dropout=config.level3_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.level3_n_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        game_embeddings: Tensor,
        game_mask: Tensor,
        identity_repr: Tensor,
    ) -> Tensor:
        """Run Level 3 attention.

        Args:
            game_embeddings: (batch, n_games, d_model)
            game_mask: (batch, n_games) bool — True=valid
            identity_repr: (batch, identity_repr_dim)

        Returns:
            (batch, d_model) — [PROJ] token output
        """
        batch_size = game_embeddings.shape[0]

        # Build [PROJ] token = learnable param + identity projection
        identity_bias = self.identity_proj(identity_repr)  # (batch, d_model)
        proj = self.proj_token.unsqueeze(0).expand(batch_size, -1) + identity_bias  # (batch, d_model)

        # Prepend [PROJ] to game sequence: (batch, 1 + n_games, d_model)
        sequence = torch.cat([proj.unsqueeze(1), game_embeddings], dim=1)

        # Build key_padding_mask: True=ignore for nn.TransformerEncoder
        # Position 0 ([PROJ]) is always valid
        proj_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=game_mask.device)
        key_padding_mask = torch.cat([proj_valid, ~game_mask], dim=1)  # (batch, 1 + n_games)

        out = self.transformer(sequence, src_key_padding_mask=key_padding_mask)
        return out[:, 0, :]  # Extract [PROJ] token


class HierarchicalPredictionHead(nn.Module):
    """Prediction head for hierarchical model.

    Input: cat(level3_output, identity_repr) of dim d_model + identity_repr_dim
    Output: (batch, n_targets)
    """

    def __init__(
        self,
        d_model: int,
        identity_repr_dim: int,
        n_targets: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = d_model + identity_repr_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict performance stats.

        Args:
            x: (batch, d_model + identity_repr_dim)

        Returns:
            (batch, n_targets)
        """
        return self.network(x)


class HierarchicalModel(nn.Module):
    """Hierarchical model composing frozen backbone + identity + Level 3 + head.

    The backbone (ContextualPerformanceModel) is frozen — all its parameters
    have requires_grad=False. Gradients flow through hidden state values to
    downstream learnable modules (identity, level3, head).
    """

    # Max samples per backbone forward pass.  The backbone builds a
    # (batch*heads, seq_len, seq_len) float attention mask — at batch=32,
    # heads=8, seq_len=2048 that is 4 GB.  Micro-batching caps this at
    # ~500 MB (micro_batch=4).
    BACKBONE_MICRO_BATCH: int = 4

    def __init__(
        self,
        backbone: ContextualPerformanceModel,
        hier_config: HierarchicalModelConfig,
        n_targets: int,
        stat_input_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.identity_module = IdentityModule(hier_config, stat_input_dim)
        self.game_pooler = GamePooler()
        self.level3 = Level3Attention(hier_config)
        self.head = HierarchicalPredictionHead(
            d_model=hier_config.level3_d_model,
            identity_repr_dim=hier_config.identity_repr_dim,
            n_targets=n_targets,
            dropout=hier_config.level3_dropout,
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    @staticmethod
    def _slice_batch(batch: TensorizedBatch, start: int, end: int) -> TensorizedBatch:
        """Slice a TensorizedBatch along the batch dimension."""
        return TensorizedBatch(
            pitch_type_ids=batch.pitch_type_ids[start:end],
            pitch_result_ids=batch.pitch_result_ids[start:end],
            bb_type_ids=batch.bb_type_ids[start:end],
            stand_ids=batch.stand_ids[start:end],
            p_throws_ids=batch.p_throws_ids[start:end],
            pa_event_ids=batch.pa_event_ids[start:end],
            numeric_features=batch.numeric_features[start:end],
            numeric_mask=batch.numeric_mask[start:end],
            padding_mask=batch.padding_mask[start:end],
            player_token_mask=batch.player_token_mask[start:end],
            game_ids=batch.game_ids[start:end],
            seq_lengths=batch.seq_lengths[start:end],
        )

    def forward(
        self,
        batch: TensorizedBatch,
        stat_features: Tensor,
        archetype_ids: Tensor,
    ) -> dict[str, Tensor]:
        """Run hierarchical model.

        Args:
            batch: TensorizedBatch from the tensorizer
            stat_features: (batch, stat_input_dim) — player stat profile features
            archetype_ids: (batch,) long — archetype cluster IDs

        Returns:
            {"performance_preds": (batch, n_targets)}
        """
        # 1. Run frozen backbone to get hidden states in micro-batches.
        # The backbone builds a (batch*heads, seq_len, seq_len) float
        # attention mask that is O(batch * heads * seq_len^2).  With
        # batch=32, heads=8, seq_len=2048 that is 4 GB.  Micro-batching
        # caps peak memory at ~500 MB.  no_grad prevents storing the
        # computation graph; detach breaks gradient flow.
        bs = batch.pitch_type_ids.shape[0]
        mb = self.BACKBONE_MICRO_BATCH
        hidden_parts: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, bs, mb):
                end = min(start + mb, bs)
                micro = self._slice_batch(batch, start, end)
                out = self.backbone(micro)
                hidden_parts.append(out["transformer_output"])
        hidden = torch.cat(hidden_parts, dim=0).detach()  # (batch, seq_len, d_model)

        # 2. Identity branch
        identity_repr = self.identity_module(stat_features, archetype_ids)

        # 3. Pool per game
        game_embeddings, game_mask = self.game_pooler(
            hidden, batch.game_ids, batch.padding_mask, batch.player_token_mask,
        )

        # 4. Level 3 attention
        proj_output = self.level3(game_embeddings, game_mask, identity_repr)

        # 5. Prediction head
        head_input = torch.cat([proj_output, identity_repr], dim=-1)
        preds = self.head(head_input)

        return {"performance_preds": preds}
