from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature


def sprint_speed_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract sprint speed from a seasonal aggregate row."""
    if not rows:
        return {"sprint_speed": float("nan")}
    row = rows[0]
    speed = row.get("sprint_speed")
    if speed is None or (isinstance(speed, float) and math.isnan(speed)):
        return {"sprint_speed": float("nan")}
    return {"sprint_speed": float(speed)}


SPRINT_SPEED_TRANSFORM = TransformFeature(
    name="sprint_speed",
    source=Source.SPRINT_SPEED,
    columns=("sprint_speed",),
    group_by=("player_id", "season"),
    transform=sprint_speed_profile,
    outputs=("sprint_speed",),
)
