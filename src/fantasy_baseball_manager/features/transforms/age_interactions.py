from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features.types import DerivedTransformFeature

_PEAK_AGE = 29


def _age_interact(age: float | None, stat: float | None) -> float:
    if age is None or stat is None or math.isnan(age) or math.isnan(stat):
        return float("nan")
    return (age - _PEAK_AGE) * stat


def age_interaction_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = rows[0]
    age = row.get("age")
    return {
        "age_avg_interact": _age_interact(age, row.get("avg_1")),
        "age_obp_interact": _age_interact(age, row.get("obp_1")),
        "age_slg_interact": _age_interact(age, row.get("slg_1")),
    }


AGE_INTERACTIONS = DerivedTransformFeature(
    name="age_interactions",
    inputs=("age", "avg_1", "obp_1", "slg_1"),
    group_by=("player_id", "season"),
    transform=age_interaction_profile,
    outputs=("age_avg_interact", "age_obp_interact", "age_slg_interact"),
)
