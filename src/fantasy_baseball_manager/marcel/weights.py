def weighted_rate(
    *,
    stats: list[float],
    opportunities: list[float],
    weights: list[float],
    league_rate: float,
    regression_pa: float,
) -> float:
    """Compute a weighted rate with regression to the league mean.

    Treats regression_pa plate appearances (or outs, for pitchers) of
    league-average performance as additional observations.
    """
    numerator = sum(w * s for w, s in zip(weights, stats, strict=False))
    denominator = sum(w * o for w, o in zip(weights, opportunities, strict=False))
    numerator += regression_pa * league_rate
    denominator += regression_pa
    return numerator / denominator


def projected_pa(*, pa_y1: float, pa_y2: float) -> float:
    """Project plate appearances for a batter."""
    return 0.5 * pa_y1 + 0.1 * pa_y2 + 200


def projected_ip(*, ip_y1: float, ip_y2: float, is_starter: bool) -> float:
    """Project innings pitched."""
    base = 0.5 * ip_y1 + 0.1 * ip_y2
    return base + (60 if is_starter else 25)
