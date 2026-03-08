"""Generate weighted playing-time scenario projections from residual percentile buckets."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain.injury_discount import (
    _BATTER_RATE_STATS,
    _PITCHER_RATE_STATS,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Projection
    from fantasy_baseball_manager.models.playing_time.engine import (
        ResidualBuckets,
        ResidualPercentiles,
    )

DEFAULT_SCENARIO_WEIGHTS: dict[int, float] = {
    10: 0.15,
    25: 0.20,
    50: 0.30,
    75: 0.20,
    90: 0.15,
}

_PERCENTILE_ATTR = {10: "p10", 25: "p25", 50: "p50", 75: "p75", 90: "p90"}

_BATTER_PT_KEY = "pa"
_PITCHER_PT_KEY = "ip"
_BATTER_PT_MAX = 750.0
_PITCHER_PT_MAX = 250.0


def _pt_key(player_type: str) -> str:
    return _BATTER_PT_KEY if player_type == "batter" else _PITCHER_PT_KEY


def _pt_max(player_type: str) -> float:
    return _BATTER_PT_MAX if player_type == "batter" else _PITCHER_PT_MAX


def _rate_stats(player_type: str) -> frozenset[str]:
    return _BATTER_RATE_STATS if player_type == "batter" else _PITCHER_RATE_STATS


def scale_projection_to_pt(projection: Projection, target_pa_or_ip: float) -> Projection:
    """Scale a projection's counting stats to a target PA/IP. Rate stats are preserved."""
    pt_key = _pt_key(projection.player_type)
    max_pt = _pt_max(projection.player_type)
    rate = _rate_stats(projection.player_type)

    clamped_target = max(0.0, min(target_pa_or_ip, max_pt))
    original_pt = projection.stat_json.get(pt_key, 0.0)

    if original_pt == 0.0:
        # Can't scale from 0 — set PT to target, counting stats stay 0
        new_stats = dict(projection.stat_json)
        new_stats[pt_key] = clamped_target
        return dataclasses.replace(projection, stat_json=new_stats)

    factor = clamped_target / original_pt

    new_stats: dict[str, object] = {}
    for key, value in projection.stat_json.items():
        if key in rate or not isinstance(value, int | float):
            new_stats[key] = value
        elif key == pt_key:
            new_stats[key] = clamped_target
        else:
            new_stats[key] = value * factor
    return dataclasses.replace(projection, stat_json=new_stats)


def generate_scenarios(
    projection: Projection,
    residual_percentiles: ResidualPercentiles,
    point_estimate_pt: float,
    scenario_weights: dict[int, float] | None = None,
) -> list[tuple[Projection, float]]:
    """Generate weighted scenario projections from residual percentiles.

    For each percentile in the weights dict, compute scenario PT as
    point_estimate_pt + residual_offset, then scale the projection to that PT.
    """
    weights = scenario_weights or DEFAULT_SCENARIO_WEIGHTS

    results: list[tuple[Projection, float]] = []
    for percentile, weight in sorted(weights.items()):
        attr = _PERCENTILE_ATTR[percentile]
        offset = getattr(residual_percentiles, attr)
        scenario_pt = point_estimate_pt + offset
        scaled = scale_projection_to_pt(projection, scenario_pt)
        results.append((scaled, weight))
    return results


def generate_pool_scenarios(
    projections: list[Projection],
    residual_buckets_map: dict[str, ResidualBuckets],
    player_bucket_keys: dict[int, str],
    scenario_weights: dict[int, float] | None = None,
) -> dict[int, list[tuple[Projection, float]]]:
    """Generate scenarios for a pool of players.

    Players with no bucket key in player_bucket_keys get a single scenario
    at point-estimate with weight 1.0 (degrades to point estimate).
    """
    result: dict[int, list[tuple[Projection, float]]] = {}

    for proj in projections:
        bucket_key = player_bucket_keys.get(proj.player_id)
        buckets = residual_buckets_map.get(proj.player_type)

        if bucket_key is None or buckets is None:
            result[proj.player_id] = [(proj, 1.0)]
            continue

        percs = buckets.buckets.get(bucket_key) or buckets.buckets.get(buckets.fallback_key)
        if percs is None:
            result[proj.player_id] = [(proj, 1.0)]
            continue

        pt_key = _pt_key(proj.player_type)
        point_pt = proj.stat_json.get(pt_key, 0.0)
        result[proj.player_id] = generate_scenarios(proj, percs, point_pt, scenario_weights)

    return result
