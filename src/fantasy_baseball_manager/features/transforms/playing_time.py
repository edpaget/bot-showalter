from __future__ import annotations

from typing import Any


def make_war_threshold_transform() -> Any:
    """Inputs: war_1 → Outputs: war_above_2, war_above_4, war_below_0."""

    def transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        war = rows[0].get("war_1") or 0.0
        return {
            "war_above_2": 1.0 if war >= 2.0 else 0.0,
            "war_above_4": 1.0 if war >= 4.0 else 0.0,
            "war_below_0": 1.0 if war < 0.0 else 0.0,
        }

    return transform


def make_starter_ratio_transform() -> Any:
    """Inputs: gs_1, g_1 → Outputs: starter_ratio."""

    def transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        gs = rows[0].get("gs_1")
        g = rows[0].get("g_1")
        if gs is None or g is None or g == 0:
            return {"starter_ratio": 0.0}
        return {"starter_ratio": gs / g}

    return transform


def make_il_severity_transform() -> Any:
    """Inputs: il_days_1 → Outputs: il_minor, il_moderate, il_severe."""

    def transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        days = rows[0].get("il_days_1") or 0
        return {
            "il_minor": 1.0 if 0 < days <= 20 else 0.0,
            "il_moderate": 1.0 if 20 < days <= 60 else 0.0,
            "il_severe": 1.0 if days > 60 else 0.0,
        }

    return transform


def make_il_summary_transform(lags: int = 3) -> Any:
    """Sum IL days across lags and compute recurrence flag.

    Inputs: il_days_1..il_days_{lags}, il_stints_1, il_stints_2
    Outputs: il_days_3yr, il_recurrence
    """

    def transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = rows[0]
        total_days = sum(row.get(f"il_days_{i}") or 0 for i in range(1, lags + 1))
        stints_1 = row.get("il_stints_1") or 0
        stints_2 = row.get("il_stints_2") or 0
        recurrence = 1.0 if stints_1 > 0 and stints_2 > 0 else 0.0
        return {"il_days_3yr": total_days, "il_recurrence": recurrence}

    return transform


def make_pt_interaction_transform() -> Any:
    """Inputs: war_1, pt_trend, age, il_recurrence → Outputs: war_trend, age_il_interact."""

    def transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = rows[0]
        war = row.get("war_1") or 0.0
        pt_trend = row.get("pt_trend") or 1.0
        age = row.get("age") or 0
        il_recurrence = row.get("il_recurrence") or 0.0
        return {
            "war_trend": war * pt_trend,
            "age_il_interact": max(0, age - 30) * il_recurrence,
        }

    return transform


def make_pt_trend_transform(pt_column: str) -> Any:
    """Compute playing time trend as ratio of year-1 to year-2.

    Inputs: {pt_column}_1, {pt_column}_2
    Outputs: pt_trend
    """

    def transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = rows[0]
        val_1 = row.get(f"{pt_column}_1")
        val_2 = row.get(f"{pt_column}_2")
        if val_1 is None or val_2 is None or val_2 == 0:
            return {"pt_trend": 1.0}
        return {"pt_trend": val_1 / val_2}

    return transform
