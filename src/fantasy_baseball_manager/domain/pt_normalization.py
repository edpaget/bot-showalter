from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain.projection_accuracy import (
    BATTING_COUNTING_STATS,
    BATTING_RATE_STATS,
    PITCHING_COUNTING_STATS,
    PITCHING_RATE_STATS,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.domain.projection import Projection

_BATTER_COUNTING_SET = set(BATTING_COUNTING_STATS)
_BATTER_RATE_SET = set(BATTING_RATE_STATS)
_PITCHER_COUNTING_SET = set(PITCHING_COUNTING_STATS)
_PITCHER_RATE_SET = set(PITCHING_RATE_STATS)


@dataclass(frozen=True)
class ConsensusLookup:
    batting_pt: dict[int, float]
    pitching_pt: dict[int, float]


def normalize_projection_pt(projection: Projection, consensus_pt: float) -> Projection:
    """Rescale counting stats to a consensus playing-time baseline.

    Rate stats are left unchanged. The PA (batters) or IP (pitchers) field
    is replaced with *consensus_pt*. If the original PT is missing or zero,
    the projection is returned unmodified.
    """
    stat_json = projection.stat_json
    is_pitcher = projection.player_type == "pitcher"

    pt_key = "ip" if is_pitcher else "pa"
    old_pt = stat_json.get(pt_key)
    if old_pt is None or old_pt == 0:
        return projection

    old_pt_f = float(old_pt)
    ratio = consensus_pt / old_pt_f

    counting_set = _PITCHER_COUNTING_SET if is_pitcher else _BATTER_COUNTING_SET
    rate_set = _PITCHER_RATE_SET if is_pitcher else _BATTER_RATE_SET

    new_stats: dict[str, Any] = {}
    for key, val in stat_json.items():
        if key == pt_key:
            new_stats[key] = consensus_pt
        elif key in counting_set:
            new_stats[key] = float(val) * ratio
        elif key in rate_set:
            new_stats[key] = val
        else:
            # Unknown / hybrid stats (e.g. war) — leave unchanged
            new_stats[key] = val

    return replace(projection, stat_json=new_stats)


def build_consensus_lookup(
    *system_projections: list[Projection],
    weights: Sequence[float] | None = None,
) -> ConsensusLookup:
    """Build a consensus playing-time lookup from N projection systems.

    For each player, computes a weighted average of PA (batters) or IP
    (pitchers) across all provided systems. Falls back to available systems
    when a player is missing from some.

    If *weights* is None, equal weights are used.
    """
    effective_weights = list(weights) if weights is not None else [1.0] * len(system_projections)

    batting_pts: dict[int, list[tuple[float, float]]] = {}
    pitching_pts: dict[int, list[tuple[float, float]]] = {}

    for system_idx, proj_list in enumerate(system_projections):
        w = effective_weights[system_idx]
        for proj in proj_list:
            if proj.player_type == "pitcher":
                ip = proj.stat_json.get("ip")
                if ip is not None:
                    pitching_pts.setdefault(proj.player_id, []).append((float(ip), w))
            else:
                pa = proj.stat_json.get("pa")
                if pa is not None:
                    batting_pts.setdefault(proj.player_id, []).append((float(pa), w))

    def _weighted_avg(entries: list[tuple[float, float]]) -> float:
        total_w = sum(w for _, w in entries)
        return sum(v * w for v, w in entries) / total_w

    batting_pt = {pid: _weighted_avg(vals) for pid, vals in batting_pts.items()}
    pitching_pt = {pid: _weighted_avg(vals) for pid, vals in pitching_pts.items()}

    return ConsensusLookup(batting_pt=batting_pt, pitching_pt=pitching_pt)
