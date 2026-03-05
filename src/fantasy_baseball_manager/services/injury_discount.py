from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Projection

_SEASON_DAYS = 183  # ~April 1 to Sept 30

_BATTER_RATE_STATS = frozenset({"avg", "obp", "slg", "ops", "iso", "babip", "woba", "wrc_plus"})
_PITCHER_RATE_STATS = frozenset({"era", "whip", "k9", "bb9", "hr9", "k_bb", "fip", "xfip"})


def apply_injury_discount(
    stat_json: dict[str, Any],
    expected_days_lost: float,
    player_type: str,
) -> dict[str, Any]:
    """Scale counting stats by max(0, 1 - expected_days_lost / 183). Preserve rate stats."""
    factor = max(0.0, 1.0 - expected_days_lost / _SEASON_DAYS)
    if factor == 1.0:
        return stat_json

    rate_stats = _BATTER_RATE_STATS if player_type == "batter" else _PITCHER_RATE_STATS

    result: dict[str, Any] = {}
    for key, value in stat_json.items():
        if key in rate_stats or not isinstance(value, int | float):
            result[key] = value
        else:
            result[key] = value * factor
    return result


def discount_projections(
    projections: list[Projection],
    injury_map: dict[int, float],
) -> list[Projection]:
    """Apply injury discount to a list of projections. Players not in injury_map are unchanged."""
    if not injury_map:
        return projections

    result: list[Projection] = []
    for proj in projections:
        days_lost = injury_map.get(proj.player_id)
        if days_lost is None:
            result.append(proj)
        else:
            adjusted = apply_injury_discount(proj.stat_json, days_lost, proj.player_type)
            result.append(dataclasses.replace(proj, stat_json=adjusted))
    return result
