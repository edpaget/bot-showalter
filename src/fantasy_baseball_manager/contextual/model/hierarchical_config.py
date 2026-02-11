"""Configuration for hierarchical model with identity conditioning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HierarchicalModelConfig:
    """Configuration for the hierarchical model (Phase 2a).

    Defines identity module, game pooler, and Level 3 attention parameters.
    The backbone (EventEmbedder + GamestateTransformer) reuses ModelConfig.
    """

    # Identity module
    identity_stat_dim: int = 64
    identity_archetype_dim: int = 32
    identity_repr_dim: int = 96  # stat + archetype
    n_archetypes: int = 20
    batter_stat_input_dim: int = 19  # 6 stats * 3 horizons + age
    pitcher_stat_input_dim: int = 13  # 4 stats * 3 horizons + age

    # Level 3 attention (game → projection)
    level3_d_model: int = 256  # match backbone d_model
    level3_n_heads: int = 4
    level3_n_layers: int = 2
    level3_ff_dim: int = 512
    level3_dropout: float = 0.1
    level3_max_games: int = 50
