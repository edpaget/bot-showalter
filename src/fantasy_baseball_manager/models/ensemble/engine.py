import math
from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.domain.projection import StatDistribution


def _weighted_percentile(sorted_values: list[float], cum_weights: list[float], percentile: float) -> float:
    """Compute a weighted percentile via linear interpolation on the CDF.

    ``sorted_values`` and ``cum_weights`` must be sorted by value.
    ``cum_weights`` holds the *cumulative* weight at each point, normalised
    so the last entry equals 1.0.  ``percentile`` is in [0, 1].
    """
    # Clamp to range
    if percentile <= cum_weights[0]:
        return sorted_values[0]
    if percentile >= cum_weights[-1]:
        return sorted_values[-1]

    for i in range(1, len(cum_weights)):
        if cum_weights[i] >= percentile:
            # Linear interpolation between i-1 and i
            lo_cdf = cum_weights[i - 1]
            hi_cdf = cum_weights[i]
            if hi_cdf == lo_cdf:
                return sorted_values[i]
            frac = (percentile - lo_cdf) / (hi_cdf - lo_cdf)
            return sorted_values[i - 1] + frac * (sorted_values[i] - sorted_values[i - 1])

    return sorted_values[-1]  # pragma: no cover


def weighted_spread(
    projections: Sequence[tuple[dict[str, Any], float]],
    stats: Sequence[str],
) -> dict[str, StatDistribution]:
    """Compute distributional spread from disagreement among projection systems.

    For each stat, builds a weighted CDF from the component values and
    extracts percentiles via linear interpolation.  Returns an empty dict
    when fewer than 2 systems contribute (no spread to measure).
    """
    result: dict[str, StatDistribution] = {}

    for stat in stats:
        # Collect (value, weight) pairs for systems that have this stat
        pairs: list[tuple[float, float]] = []
        for stat_json, weight in projections:
            if stat in stat_json:
                pairs.append((float(stat_json[stat]), weight))

        if len(pairs) < 2:
            continue

        # Sort by value
        pairs.sort(key=lambda p: p[0])
        values = [v for v, _ in pairs]
        weights = [w for _, w in pairs]

        # Build normalised cumulative weights using midpoint convention
        total_weight = sum(weights)
        cum_weights: list[float] = []
        running = 0.0
        for w in weights:
            # Midpoint of this item's weight band in the CDF
            running += w
            cum_weights.append((running - w / 2) / total_weight)

        # Compute percentiles
        p10 = _weighted_percentile(values, cum_weights, 0.10)
        p25 = _weighted_percentile(values, cum_weights, 0.25)
        p50 = _weighted_percentile(values, cum_weights, 0.50)
        p75 = _weighted_percentile(values, cum_weights, 0.75)
        p90 = _weighted_percentile(values, cum_weights, 0.90)

        # Weighted mean and std
        w_mean = sum(v * w for v, w in zip(values, weights)) / total_weight
        w_var = sum(w * (v - w_mean) ** 2 for v, w in zip(values, weights)) / total_weight
        w_std = math.sqrt(w_var)

        result[stat] = StatDistribution(
            stat=stat,
            p10=p10,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            mean=w_mean,
            std=w_std,
        )

    return result


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
    consensus_pt: float | None = None,
) -> dict[str, float]:
    """Average rates across systems (weighted), average PT, return both.

    Rates are weight-averaged directly. When *consensus_pt* is ``None``
    (the default), the playing-time stat is also weight-averaged.  When a
    consensus value is provided, rates are weight-averaged but the PT stat
    is set to the consensus value directly.
    """
    if not projections:
        return {}

    if consensus_pt is not None:
        result = weighted_average(projections, stats=rate_stats)
        result[pt_stat] = consensus_pt
        return result

    all_stats = [*rate_stats, pt_stat]
    return weighted_average(projections, stats=all_stats)
