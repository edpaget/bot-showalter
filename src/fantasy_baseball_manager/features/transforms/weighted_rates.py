from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.features.types import RowTransform


def make_weighted_rates_transform(
    categories: Sequence[str],
    weights: tuple[float, ...],
    pt_column: str,
) -> RowTransform:
    """Factory that builds a weighted-average-rates transform.

    The returned transform reads lag columns ``{cat}_1 .. {cat}_N`` and
    ``{pt_column}_1 .. {pt_column}_N`` from a single-row group, then computes:

    * ``{cat}_wavg = sum(stat_i * w_i) / sum(pt_i * w_i)`` for each category
    * ``weighted_pt = sum(pt_i * w_i)``
    """
    n = len(weights)
    cats = list(categories)

    def _transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = rows[0] if rows else {}
        weighted_stats: dict[str, float] = {cat: 0.0 for cat in cats}
        weighted_pt = 0.0

        for i in range(n):
            lag = i + 1
            pt_val = row.get(f"{pt_column}_{lag}")
            if pt_val is None:
                continue
            pt = float(pt_val)
            w = weights[i]
            weighted_pt += pt * w
            for cat in cats:
                stat_val = row.get(f"{cat}_{lag}")
                stat = float(stat_val) if stat_val is not None else 0.0
                weighted_stats[cat] += stat * w

        result: dict[str, Any] = {}
        for cat in cats:
            if weighted_pt == 0.0:
                result[f"{cat}_wavg"] = 0.0
            else:
                result[f"{cat}_wavg"] = weighted_stats[cat] / weighted_pt
        result["weighted_pt"] = weighted_pt
        return result

    return _transform
