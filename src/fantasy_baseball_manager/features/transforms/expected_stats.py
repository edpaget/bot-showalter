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


def expected_stats_advanced_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute advanced xwOBA distribution features from Statcast estimates.

    Outputs:
      xwoba_elite_rate — fraction of batted balls with xwOBA >= 1.0
      xwoba_fastball   — average xwOBA on fastballs (FF, SI)
    """
    nan = float("nan")

    xwoba_valid = [r for r in rows if r.get("estimated_woba_using_speedangle") is not None]
    n = len(xwoba_valid)
    if n == 0:
        return {"xwoba_elite_rate": nan, "xwoba_fastball": nan}

    elite_count = sum(1 for r in xwoba_valid if r["estimated_woba_using_speedangle"] >= 1.0)
    xwoba_elite_rate = elite_count / n

    fb_rows = [r for r in xwoba_valid if r.get("pitch_type") in ("FF", "SI")]
    fb_n = len(fb_rows)
    xwoba_fastball = sum(r["estimated_woba_using_speedangle"] for r in fb_rows) / fb_n if fb_n > 0 else nan

    return {"xwoba_elite_rate": xwoba_elite_rate, "xwoba_fastball": xwoba_fastball}


EXPECTED_STATS_ADVANCED = TransformFeature(
    name="expected_stats_advanced",
    source=Source.STATCAST,
    columns=("estimated_woba_using_speedangle", "pitch_type"),
    group_by=("player_id", "season"),
    transform=expected_stats_advanced_profile,
    outputs=("xwoba_elite_rate", "xwoba_fastball"),
)
