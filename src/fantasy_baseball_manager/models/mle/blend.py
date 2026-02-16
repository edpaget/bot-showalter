"""Blend MLE translations with MLB stats using reliability-weighted regression."""

from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.models.mle.types import BlendConfig, BlendedStatLine


def blend_rate(
    *,
    mlb_pa: int,
    mlb_rate: float,
    mle_pa: int,
    mle_rate: float,
    discount: float,
    regression_pa: float,
    league_rate: float,
) -> float:
    """Blend a single rate component across MLB, MLE, and league average.

    Formula: (mlb_pa * mlb_rate + mle_pa * discount * mle_rate + regression_pa * league_rate)
           / (mlb_pa + mle_pa * discount + regression_pa)
    """
    effective_mle = mle_pa * discount
    numerator = mlb_pa * mlb_rate + effective_mle * mle_rate + regression_pa * league_rate
    denominator = mlb_pa + effective_mle + regression_pa
    return numerator / denominator


def blend_mle_with_mlb(
    *,
    mlb_seasons: Sequence[dict[str, float]],
    mle_stats: dict[str, Any],
    config: BlendConfig,
    league_rates: dict[str, float],
    player_id: int = 0,
    season: int = 0,
) -> BlendedStatLine:
    """Blend MLE translation rates with MLB season stats.

    mlb_seasons: list of dicts with 'pa', 'k_pct', 'bb_pct', 'iso', 'babip'.
    mle_stats: MLE projection stat_json with 'pa', 'k_pct', 'bb_pct', 'iso', 'babip'.
    """
    # Sum MLB PA and compute PA-weighted rates
    mlb_pa = int(sum(s["pa"] for s in mlb_seasons))
    mlb_rates: dict[str, float] = {}
    for component in ("k_pct", "bb_pct", "iso", "babip"):
        if mlb_pa > 0:
            mlb_rates[component] = sum(s["pa"] * s[component] for s in mlb_seasons) / mlb_pa
        else:
            mlb_rates[component] = 0.0

    mle_pa = int(float(mle_stats.get("pa", 0)))
    effective_pa = mlb_pa + mle_pa * config.discount_factor

    # Blend each component with its specific stabilization rate
    rates: dict[str, float] = {}
    for component in ("k_pct", "bb_pct", "iso", "babip"):
        stab = config.stabilization_pa.get(component, 120.0)
        mle_rate = float(mle_stats.get(component, 0.0))
        rates[component] = blend_rate(
            mlb_pa=mlb_pa,
            mlb_rate=mlb_rates[component],
            mle_pa=mle_pa,
            mle_rate=mle_rate,
            discount=config.discount_factor,
            regression_pa=stab,
            league_rate=league_rates.get(component, 0.0),
        )

    return BlendedStatLine(
        player_id=player_id,
        season=season,
        mlb_pa=mlb_pa,
        mle_pa=mle_pa,
        effective_pa=effective_pa,
        rates=rates,
    )
