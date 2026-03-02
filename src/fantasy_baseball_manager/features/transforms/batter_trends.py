from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features.types import DerivedTransformFeature


def _sub(a: float | None, b: float | None) -> float:
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return float("nan")
    return a - b


def batter_trend_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[0]
    return {
        "avg_trend": _sub(row.get("avg_1"), row.get("avg_2")),
        "obp_trend": _sub(row.get("obp_1"), row.get("obp_2")),
        "slg_trend": _sub(row.get("slg_1"), row.get("slg_2")),
    }


def batter_stability_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[0]
    avg_trend = _sub(row.get("avg_1"), row.get("avg_2"))
    obp_trend = _sub(row.get("obp_1"), row.get("obp_2"))
    slg_trend = _sub(row.get("slg_1"), row.get("slg_2"))
    return {
        "avg_stability": float("nan") if math.isnan(avg_trend) else abs(avg_trend),
        "obp_stability": float("nan") if math.isnan(obp_trend) else abs(obp_trend),
        "slg_stability": float("nan") if math.isnan(slg_trend) else abs(slg_trend),
    }


BATTER_TRENDS = DerivedTransformFeature(
    name="batter_trends",
    inputs=("avg_1", "avg_2", "obp_1", "obp_2", "slg_1", "slg_2"),
    group_by=("player_id", "season"),
    transform=batter_trend_profile,
    outputs=("avg_trend", "obp_trend", "slg_trend"),
)

BATTER_STABILITY = DerivedTransformFeature(
    name="batter_stability",
    inputs=("avg_1", "avg_2", "obp_1", "obp_2", "slg_1", "slg_2"),
    group_by=("player_id", "season"),
    transform=batter_stability_profile,
    outputs=("avg_stability", "obp_stability", "slg_stability"),
)
