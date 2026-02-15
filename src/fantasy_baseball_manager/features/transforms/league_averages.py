from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.features.types import RowTransform


def make_league_avg_transform(
    categories: Sequence[str],
    pt_column: str,
) -> RowTransform:
    """Factory that builds a league-average-rates transform.

    The returned transform receives **all rows for a season** (grouped by
    ``("season",)``).  It reads ``{cat}_1`` and ``{pt_column}_1`` from each
    row (lag-1 = most-recent year), then computes:

    * ``league_{cat}_rate = sum({cat}_1) / sum({pt}_1)`` for each category.
    """
    cats = list(categories)

    def _transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total_stats: dict[str, float] = {cat: 0.0 for cat in cats}
        total_pt = 0.0

        for row in rows:
            pt_val = row.get(f"{pt_column}_1")
            if pt_val is None:
                continue
            pt = float(pt_val)
            if pt == 0.0:
                continue
            total_pt += pt
            for cat in cats:
                stat_val = row.get(f"{cat}_1")
                total_stats[cat] += float(stat_val) if stat_val is not None else 0.0

        result: dict[str, Any] = {}
        for cat in cats:
            if total_pt == 0.0:
                result[f"league_{cat}_rate"] = 0.0
            else:
                result[f"league_{cat}_rate"] = total_stats[cat] / total_pt
        return result

    return _transform
