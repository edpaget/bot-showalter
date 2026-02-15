from __future__ import annotations

from typing import Any

from fantasy_baseball_manager.features.types import Source, TransformFeature

# Typical league SLG / league wOBA ratio (~0.400 / ~0.320 ≈ 1.25).
# Used to estimate xSLG from xwOBA.
_XSLG_SCALE = 1.25


def expected_stats_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute expected stats (xBA, xwOBA, xSLG) from Statcast estimates."""
    valid = [
        r
        for r in rows
        if r.get("estimated_ba_using_speedangle") is not None and r.get("estimated_woba_using_speedangle") is not None
    ]
    n = len(valid)
    if n == 0:
        return {"xba": 0.0, "xwoba": 0.0, "xslg": 0.0}

    total_xba = sum(r["estimated_ba_using_speedangle"] for r in valid)
    total_xwoba = sum(r["estimated_woba_using_speedangle"] for r in valid)

    xba = total_xba / n
    xwoba = total_xwoba / n
    xslg = xwoba * _XSLG_SCALE

    return {"xba": xba, "xwoba": xwoba, "xslg": xslg}


EXPECTED_STATS = TransformFeature(
    name="expected_stats",
    source=Source.STATCAST,
    columns=("estimated_ba_using_speedangle", "estimated_woba_using_speedangle"),
    group_by=("player_id", "season"),
    transform=expected_stats_profile,
    outputs=("xba", "xwoba", "xslg"),
)
