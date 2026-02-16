from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature


def expected_stats_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute expected stats (xBA, xwOBA, xSLG) from Statcast estimates."""
    valid = [
        r
        for r in rows
        if r.get("estimated_ba_using_speedangle") is not None and r.get("estimated_woba_using_speedangle") is not None
    ]
    n = len(valid)
    if n == 0:
        return {"xba": float("nan"), "xwoba": float("nan"), "xslg": float("nan")}

    total_xba = sum(r["estimated_ba_using_speedangle"] for r in valid)
    total_xwoba = sum(r["estimated_woba_using_speedangle"] for r in valid)

    xba = total_xba / n
    xwoba = total_xwoba / n

    slg_valid = [r for r in valid if r.get("estimated_slg_using_speedangle") is not None]
    slg_n = len(slg_valid)
    xslg = sum(r["estimated_slg_using_speedangle"] for r in slg_valid) / slg_n if slg_n > 0 else float("nan")

    return {"xba": xba, "xwoba": xwoba, "xslg": xslg}


EXPECTED_STATS = TransformFeature(
    name="expected_stats",
    source=Source.STATCAST,
    columns=("estimated_ba_using_speedangle", "estimated_woba_using_speedangle", "estimated_slg_using_speedangle"),
    group_by=("player_id", "season"),
    transform=expected_stats_profile,
    outputs=("xba", "xwoba", "xslg"),
)
