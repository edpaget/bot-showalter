from dataclasses import dataclass, replace
from typing import Any

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.projection_accuracy import (
    BATTING_COUNTING_STATS,
    BATTING_RATE_STATS,
    PITCHING_COUNTING_STATS,
    PITCHING_RATE_STATS,
)

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
    steamer_projections: list[Projection],
    zips_projections: list[Projection],
) -> ConsensusLookup:
    """Build a consensus playing-time lookup from two projection systems.

    For each player, averages PA (batters) or IP (pitchers) from both systems.
    Falls back to the single available system when only one is present.
    """
    batting_pts: dict[int, list[float]] = {}
    pitching_pts: dict[int, list[float]] = {}

    for proj in (*steamer_projections, *zips_projections):
        if proj.player_type == "pitcher":
            ip = proj.stat_json.get("ip")
            if ip is not None:
                pitching_pts.setdefault(proj.player_id, []).append(float(ip))
        else:
            pa = proj.stat_json.get("pa")
            if pa is not None:
                batting_pts.setdefault(proj.player_id, []).append(float(pa))

    batting_pt = {pid: sum(vals) / len(vals) for pid, vals in batting_pts.items()}
    pitching_pt = {pid: sum(vals) / len(vals) for pid, vals in pitching_pts.items()}

    return ConsensusLookup(batting_pt=batting_pt, pitching_pt=pitching_pt)
