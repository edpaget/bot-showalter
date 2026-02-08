from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from fantasy_baseball_manager.player.identity import Player


class PlayerMetadata(TypedDict, total=False):
    """Type-safe metadata for PlayerRates.

    All fields are optional since metadata is populated incrementally
    by different pipeline stages.
    """

    # Input metadata (set by rate computers)
    pa_per_year: list[float]
    ip_per_year: list[float] | float
    games_per_year: list[float]
    is_starter: bool
    position: str
    team: str
    avg_league_rates: dict[str, float]
    target_rates: dict[str, float]

    # Platoon split metadata
    rates_vs_lhp: dict[str, float]
    rates_vs_rhp: dict[str, float]
    pct_vs_lhp: float
    pct_vs_rhp: float

    # Enhanced playing time metadata
    injury_factor: float
    age_pt_factor: float
    volatility_factor: float
    base_pa: float
    base_ip: float

    # Pitcher normalization metadata
    observed_babip: float
    expected_babip: float
    expected_lob_pct: float

    # Batter BABIP adjuster metadata
    x_babip: float
    observed_batter_babip: float
    babip_singles_adjustment: float

    # Pitcher BABIP skill adjuster metadata
    pitcher_x_babip: float
    pitcher_gb_pct: float
    pitcher_babip_skill_blended: float

    # Statcast adjuster metadata (batters)
    statcast_blended: bool
    statcast_xwoba: float
    blend_weight_used: float

    # Pitcher statcast adjuster metadata
    pitcher_xera: float
    pitcher_xba_against: float
    pitcher_statcast_blended: bool
    pitcher_h_blend_weight: float
    pitcher_er_blend_weight: float

    # GB residual adjuster metadata
    gb_residual_adjustments: dict[str, float]

    # Skill change adjuster metadata
    skill_change_adjustments: dict[str, float]

    # MTL rate computer/blender metadata
    mtl_predicted: bool
    marcel_rates: dict[str, float]
    mtl_blended: bool
    mtl_blend_weight: float
    mtl_rates: dict[str, float]

    # MLE rate computer metadata
    mle_applied: bool
    mle_source_level: str
    mle_source_pa: int
    mle_rates: dict[str, float]

    # MLE augmented rate computer metadata
    mle_augmented: bool
    delegate_rates: dict[str, float]

    # Contextual rate computer metadata
    contextual_predicted: bool
    contextual_games_used: int


@dataclass
class PlayerRates:
    player_id: str
    name: str
    year: int
    age: int
    rates: dict[str, float] = field(default_factory=dict)
    opportunities: float = 0.0
    metadata: PlayerMetadata = field(default_factory=dict)
    player: Player | None = field(default=None, repr=False)
