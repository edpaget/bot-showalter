from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature

_NAN = float("nan")

_SPRAY_THRESHOLD = 15.0


def batter_context_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute batter context features: avg vertical break faced and pulled fly ball rate."""
    # avg_vert_break_faced: mean of pfx_z over non-null pitches
    pfx_values = [r["pfx_z"] for r in rows if r.get("pfx_z") is not None]
    avg_vert_break = sum(pfx_values) / len(pfx_values) if pfx_values else _NAN

    # pull_fb_rate: fraction of fly balls (25 <= launch_angle < 50) that are pulled
    fly_balls = [
        r
        for r in rows
        if r.get("launch_angle") is not None
        and 25.0 <= r["launch_angle"] < 50.0
        and r.get("hc_x") is not None
        and r.get("hc_y") is not None
        and r.get("stand") is not None
    ]
    if fly_balls:
        pulled = 0
        for r in fly_balls:
            angle = math.atan2(r["hc_x"] - 125.42, 198.27 - r["hc_y"]) * 180.0 / math.pi
            if (r["stand"] == "R" and angle < -_SPRAY_THRESHOLD) or (r["stand"] != "R" and angle > _SPRAY_THRESHOLD):
                pulled += 1
        pull_fb_rate = pulled / len(fly_balls)
    else:
        pull_fb_rate = _NAN

    return {
        "avg_vert_break_faced": avg_vert_break,
        "pull_fb_rate": pull_fb_rate,
    }


BATTER_CONTEXT = TransformFeature(
    name="batter_context",
    source=Source.STATCAST,
    columns=("pfx_z", "launch_angle", "hc_x", "hc_y", "stand"),
    group_by=("player_id", "season"),
    transform=batter_context_profile,
    outputs=("avg_vert_break_faced", "pull_fb_rate"),
)
