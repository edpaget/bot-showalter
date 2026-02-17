from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature

_NAN = float("nan")

_EMPTY_RESULT: dict[str, Any] = {
    "avg_exit_velo": _NAN,
    "max_exit_velo": _NAN,
    "avg_launch_angle": _NAN,
    "barrel_pct": _NAN,
    "hard_hit_pct": _NAN,
    "gb_pct": _NAN,
    "fb_pct": _NAN,
    "ld_pct": _NAN,
    "sweet_spot_pct": _NAN,
    "exit_velo_p90": _NAN,
}


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute a percentile from a pre-sorted list using linear interpolation."""
    n = len(sorted_values)
    if n == 0:
        return _NAN
    if n == 1:
        return sorted_values[0]
    rank = pct / 100.0 * (n - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def batted_ball_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute batted-ball profile metrics from statcast pitch data."""
    batted = [r for r in rows if r.get("launch_speed") is not None]
    n = len(batted)
    if n == 0:
        return dict(_EMPTY_RESULT)

    total_velo = sum(r["launch_speed"] for r in batted)
    max_velo = max(r["launch_speed"] for r in batted)
    with_angle = [r for r in batted if r.get("launch_angle") is not None]
    barrels = sum(1 for r in batted if r.get("barrel") == 1)
    hard_hits = sum(1 for r in batted if r["launch_speed"] >= 95.0)

    avg_angle = sum(r["launch_angle"] for r in with_angle) / len(with_angle) if with_angle else _NAN

    n_angle = len(with_angle)
    if n_angle > 0:
        gb = sum(1 for r in with_angle if r["launch_angle"] < 10.0)
        fb = sum(1 for r in with_angle if 25.0 <= r["launch_angle"] < 50.0)
        ld = sum(1 for r in with_angle if 10.0 <= r["launch_angle"] < 25.0)
        sweet = sum(1 for r in with_angle if 8.0 <= r["launch_angle"] <= 32.0)
        gb_pct = gb / n_angle * 100.0
        fb_pct = fb / n_angle * 100.0
        ld_pct = ld / n_angle * 100.0
        sweet_spot_pct = sweet / n_angle * 100.0
    else:
        gb_pct = fb_pct = ld_pct = sweet_spot_pct = _NAN

    sorted_velos = sorted(r["launch_speed"] for r in batted)
    exit_velo_p90 = _percentile(sorted_velos, 90.0)

    return {
        "avg_exit_velo": total_velo / n,
        "max_exit_velo": max_velo,
        "avg_launch_angle": avg_angle,
        "barrel_pct": barrels / n * 100.0,
        "hard_hit_pct": hard_hits / n * 100.0,
        "gb_pct": gb_pct,
        "fb_pct": fb_pct,
        "ld_pct": ld_pct,
        "sweet_spot_pct": sweet_spot_pct,
        "exit_velo_p90": exit_velo_p90,
    }


_EMPTY_AGAINST_RESULT: dict[str, Any] = {
    "gb_pct_against": _NAN,
    "fb_pct_against": _NAN,
    "avg_exit_velo_against": _NAN,
    "barrel_pct_against": _NAN,
}


def batted_ball_against_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute batted-ball-against metrics for pitchers from statcast pitch data."""
    batted = [r for r in rows if r.get("launch_speed") is not None]
    n = len(batted)
    if n == 0:
        return dict(_EMPTY_AGAINST_RESULT)

    total_velo = sum(r["launch_speed"] for r in batted)
    barrels = sum(1 for r in batted if r.get("barrel") == 1)

    with_angle = [r for r in batted if r.get("launch_angle") is not None]
    n_angle = len(with_angle)
    if n_angle > 0:
        gb = sum(1 for r in with_angle if r["launch_angle"] < 10.0)
        fb = sum(1 for r in with_angle if 25.0 <= r["launch_angle"] < 50.0)
        gb_pct = gb / n_angle * 100.0
        fb_pct = fb / n_angle * 100.0
    else:
        gb_pct = fb_pct = _NAN

    return {
        "gb_pct_against": gb_pct,
        "fb_pct_against": fb_pct,
        "avg_exit_velo_against": total_velo / n,
        "barrel_pct_against": barrels / n * 100.0,
    }


BATTED_BALL = TransformFeature(
    name="batted_ball",
    source=Source.STATCAST,
    columns=("launch_speed", "launch_angle", "barrel"),
    group_by=("player_id", "season"),
    transform=batted_ball_profile,
    outputs=(
        "avg_exit_velo",
        "max_exit_velo",
        "avg_launch_angle",
        "barrel_pct",
        "hard_hit_pct",
        "gb_pct",
        "fb_pct",
        "ld_pct",
        "sweet_spot_pct",
        "exit_velo_p90",
    ),
)

BATTED_BALL_AGAINST = TransformFeature(
    name="batted_ball_against",
    source=Source.STATCAST,
    columns=("launch_speed", "launch_angle", "barrel"),
    group_by=("player_id", "season"),
    transform=batted_ball_against_profile,
    outputs=(
        "gb_pct_against",
        "fb_pct_against",
        "avg_exit_velo_against",
        "barrel_pct_against",
    ),
)
