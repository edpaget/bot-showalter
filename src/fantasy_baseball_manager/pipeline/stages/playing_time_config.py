"""Configuration for enhanced playing time projections."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlayingTimeConfig:
    """Configuration for enhanced playing time adjustments.

    Controls injury proxy penalties, age-based decline, volatility
    adjustments, and role-based caps on projected playing time.
    """

    # Injury proxy settings (conservative - only penalize severely injured)
    games_played_weight: float = 0.08  # small penalty for missed games
    min_games_pct: float = 0.40  # below this (~65 games), apply max penalty

    # Age-based PT decline (separate from performance aging)
    age_decline_start: int = 35
    age_decline_rate: float = 0.01  # 1% per year after threshold

    # Volatility adjustment
    volatility_threshold: float = 0.25  # >25% year-over-year swing
    volatility_penalty: float = 0.0  # disabled - not predictive

    # Role caps (realistic maximums)
    batter_pa_cap: int = 720
    starter_ip_cap: int = 220
    reliever_ip_cap: int = 90
    catcher_pa_cap: int = 580

    # Position-based baseline adjustments
    catcher_baseline_reduction: float = 0.15  # catchers play ~15% fewer games
