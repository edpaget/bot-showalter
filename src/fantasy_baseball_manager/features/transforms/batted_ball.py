from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature


def batted_ball_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute batted-ball profile metrics from statcast pitch data."""
    batted = [r for r in rows if r.get("launch_speed") is not None]
    n = len(batted)
    if n == 0:
        return {
            "avg_exit_velo": float("nan"),
            "max_exit_velo": float("nan"),
            "avg_launch_angle": float("nan"),
            "barrel_pct": float("nan"),
            "hard_hit_pct": float("nan"),
        }

    total_velo = sum(r["launch_speed"] for r in batted)
    max_velo = max(r["launch_speed"] for r in batted)
    total_angle = sum(r["launch_angle"] for r in batted)
    barrels = sum(1 for r in batted if r.get("barrel"))
    hard_hits = sum(1 for r in batted if r["launch_speed"] >= 95.0)

    return {
        "avg_exit_velo": total_velo / n,
        "max_exit_velo": max_velo,
        "avg_launch_angle": total_angle / n,
        "barrel_pct": barrels / n * 100.0,
        "hard_hit_pct": hard_hits / n * 100.0,
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
    ),
)
