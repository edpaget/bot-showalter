from collections.abc import Sequence
from typing import Any


def weighted_average(
    projections: Sequence[tuple[dict[str, Any], float]],
    stats: Sequence[str],
) -> dict[str, float]:
    """Weighted average of stat values across systems.

    For each stat, computes sum(value_i * weight_i) / sum(weight_i) across
    systems that have the stat. Systems missing a stat are excluded from
    that stat's calculation.
    """
    if not projections:
        return {}

    result: dict[str, float] = {}
    for stat in stats:
        numerator = 0.0
        denominator = 0.0
        for stat_json, weight in projections:
            if stat in stat_json:
                numerator += stat_json[stat] * weight
                denominator += weight
        if denominator > 0:
            result[stat] = numerator / denominator
    return result


def blend_rates(
    projections: Sequence[tuple[dict[str, Any], float]],
    rate_stats: Sequence[str],
    pt_stat: str,
) -> dict[str, float]:
    """Average rates across systems (weighted), average PT, return both.

    Rates are weight-averaged directly. The playing-time stat (pt_stat) is
    also weight-averaged. Systems missing a rate stat are excluded from
    that stat's average.
    """
    if not projections:
        return {}

    all_stats = [*rate_stats, pt_stat]
    return weighted_average(projections, stats=all_stats)
