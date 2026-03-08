from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain.injury_discount import (
    _BATTER_RATE_STATS,
    _PITCHER_RATE_STATS,
    _SEASON_DAYS,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Projection, ReplacementProfile

_VOLUME_STATS = frozenset({"pa", "ab", "ip", "g", "gs"})


def blend_stat_line(
    stat_json: dict[str, Any],
    replacement_stats: dict[str, float],
    expected_days_lost: float,
    player_type: str,
) -> dict[str, Any]:
    """Blend a player's stat line with replacement-level stats based on missed time."""
    healthy_frac = max(0.0, 1.0 - expected_days_lost / _SEASON_DAYS)
    if healthy_frac == 1.0:
        return stat_json

    missed_frac = 1.0 - healthy_frac
    rate_stats = _BATTER_RATE_STATS if player_type == "batter" else _PITCHER_RATE_STATS

    result: dict[str, Any] = {}
    for key, value in stat_json.items():
        if not isinstance(value, int | float) or key in _VOLUME_STATS:
            result[key] = value
        elif key in rate_stats:
            repl = replacement_stats.get(key, value)
            result[key] = value * healthy_frac + repl * missed_frac
        else:
            repl = replacement_stats.get(key, 0.0)
            result[key] = value * healthy_frac + repl * missed_frac
    return result


def blend_projections(
    projections: list[Projection],
    replacement_profiles: dict[str, ReplacementProfile],
    injury_map: dict[int, float],
    position_map: dict[int, list[str]],
) -> list[Projection]:
    """Apply replacement-padded blending to projections based on injury risk."""
    if not injury_map:
        return projections

    result: list[Projection] = []
    for proj in projections:
        days_lost = injury_map.get(proj.player_id)
        if days_lost is None:
            result.append(proj)
            continue

        positions = position_map.get(proj.player_id, [])
        profile: ReplacementProfile | None = None
        for pos in positions:
            if pos in replacement_profiles:
                profile = replacement_profiles[pos]
                break

        if profile is None:
            result.append(proj)
            continue

        blended = blend_stat_line(proj.stat_json, profile.stat_line, days_lost, proj.player_type)
        result.append(dataclasses.replace(proj, stat_json=blended))
    return result
