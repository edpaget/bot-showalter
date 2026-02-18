from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features.types import DerivedTransformFeature


def _mul(a: float | None, b: float | None) -> float:
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return float("nan")
    return a * b


def batted_ball_interaction_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[0]
    barrel_pct = row.get("barrel_pct")
    avg_la = row.get("avg_launch_angle")
    hard_hit = row.get("hard_hit_pct")
    pull = row.get("pull_pct")
    exit_velo = row.get("avg_exit_velo")
    zone_contact = row.get("zone_contact_pct")

    return {
        "barrel_la_interaction": _mul(barrel_pct, avg_la),
        "hard_pull_interaction": _mul(hard_hit, pull),
        "quality_contact_interaction": _mul(exit_velo, zone_contact),
    }


BATTED_BALL_INTERACTIONS = DerivedTransformFeature(
    name="batted_ball_interactions",
    inputs=(
        "barrel_pct",
        "avg_launch_angle",
        "hard_hit_pct",
        "pull_pct",
        "avg_exit_velo",
        "zone_contact_pct",
    ),
    group_by=("player_id", "season"),
    transform=batted_ball_interaction_profile,
    outputs=(
        "barrel_la_interaction",
        "hard_pull_interaction",
        "quality_contact_interaction",
    ),
)
