from __future__ import annotations

from typing import Any


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
