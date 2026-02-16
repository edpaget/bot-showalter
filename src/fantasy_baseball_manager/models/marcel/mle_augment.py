"""Augment Marcel inputs with MLE projection data."""

from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.types import MarcelInput, SeasonLine

# Categories present in MLE stat_json (per-PA counting stats)
MLE_CATEGORIES: frozenset[str] = frozenset({"h", "doubles", "triples", "hr", "bb", "so"})

BASELINE_PA: int = 200


def _extract_mle_rates(
    stat_json: dict[str, Any],
    categories: Sequence[str],
    league_rates: dict[str, float],
) -> tuple[dict[str, float], int]:
    """Extract per-PA rates from MLE stat_json for available categories.

    Returns (rates_dict, pa). Categories not in MLE data get league rate.
    """
    pa = int(float(stat_json.get("pa", 0)))
    if pa <= 0:
        return {}, 0

    rates: dict[str, float] = {}
    for cat in categories:
        if cat in stat_json and cat in MLE_CATEGORIES:
            rates[cat] = float(stat_json[cat]) / pa
        else:
            rates[cat] = league_rates.get(cat, 0.0)

    return rates, pa


def _merge_rates(
    existing_rates: dict[str, float],
    existing_pt: float,
    mle_rates: dict[str, float],
    effective_mle_pa: float,
) -> dict[str, float]:
    """PA-weight merge MLE rates into existing weighted rates."""
    total_pt = existing_pt + effective_mle_pa
    merged: dict[str, float] = {}
    for cat in existing_rates:
        existing_val = existing_rates[cat] * existing_pt
        mle_val = mle_rates.get(cat, existing_rates[cat]) * effective_mle_pa
        merged[cat] = (existing_val + mle_val) / total_pt
    return merged


def augment_inputs_with_mle(
    *,
    inputs: dict[int, MarcelInput],
    mle_projections: Sequence[Projection],
    categories: Sequence[str],
    league_rates: dict[str, float],
    discount_factor: float = 0.55,
) -> dict[int, MarcelInput]:
    """Augment Marcel inputs with MLE projection data.

    For mixed players (existing MLB + MLE): PA-weight merges MLE rates.
    For MLE-only players: creates synthetic MarcelInput with MLE rates.
    """
    result = dict(inputs)

    for proj in mle_projections:
        if proj.player_type != "batter":
            continue

        mle_rates, mle_pa = _extract_mle_rates(proj.stat_json, categories, league_rates)
        if mle_pa <= 0:
            continue

        effective_mle_pa = mle_pa * discount_factor
        player_id = proj.player_id

        if player_id in result:
            # Mixed player: merge MLE into existing
            existing = result[player_id]
            merged_rates = _merge_rates(
                existing.weighted_rates,
                existing.weighted_pt,
                mle_rates,
                effective_mle_pa,
            )
            result[player_id] = MarcelInput(
                weighted_rates=merged_rates,
                weighted_pt=existing.weighted_pt + effective_mle_pa,
                league_rates=existing.league_rates,
                age=existing.age,
                seasons=existing.seasons,
            )
        else:
            # MLE-only player: create synthetic input
            baseline_season = SeasonLine(
                stats={cat: mle_rates[cat] * BASELINE_PA for cat in categories},
                pa=BASELINE_PA,
            )
            result[player_id] = MarcelInput(
                weighted_rates=mle_rates,
                weighted_pt=effective_mle_pa,
                league_rates=dict(league_rates),
                age=0,
                seasons=(baseline_season,),
            )

    return result
