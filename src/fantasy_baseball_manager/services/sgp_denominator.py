from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    DenominatorMethod,
    Direction,
    SgpDenominators,
    SgpSeasonDenominator,
    StatType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.domain import CategoryConfig, TeamSeasonStats


def _mean_gap_denominator(values: list[float]) -> float:
    """Compute denominator using the mean-gap (Art McGee 1997) method."""
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return mean(gaps)


def _assign_ranks(values: list[float]) -> list[float]:
    """Assign 1-based ranks to sorted values, using mean rank for ties."""
    n = len(values)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[j + 1] == values[j]:
            j += 1
        # positions i..j are tied; assign mean rank (1-based)
        mean_rank = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[k] = mean_rank
        i = j + 1
    return ranks


def _regression_slope(x: list[float], y: list[float]) -> float:
    """Compute OLS slope: Σ[(x-x̄)(y-ȳ)] / Σ[(x-x̄)²]."""
    n = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n
    numerator = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y, strict=True))
    denominator = sum((xi - x_bar) ** 2 for xi in x)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _regression_denominator(values: list[float]) -> float:
    """Compute denominator using regression-slope method.

    Fits a line through (standings_points, category_value) and returns the slope.
    Standings points are 1-based ranks (1 = worst, N = best). Ties get mean rank.
    """
    ranks = _assign_ranks(values)
    return _regression_slope(ranks, values)


def compute_sgp_denominators(
    standings: list[TeamSeasonStats],
    categories: Sequence[CategoryConfig],
    *,
    method: DenominatorMethod = DenominatorMethod.MEAN_GAP,
) -> SgpDenominators:
    """Compute SGP denominators from league standings data.

    For each category and season, sorts teams by stat value and computes the
    denominator using the specified method:
    - MEAN_GAP: average gap between adjacent teams (Art McGee 1997)
    - REGRESSION: slope of linear regression through (rank, value) points

    For "lower is better" categories (ERA, WHIP), the denominator is negated.
    """
    by_season: dict[int, list[TeamSeasonStats]] = defaultdict(list)
    for team in standings:
        by_season[team.season].append(team)

    per_season: list[SgpSeasonDenominator] = []

    compute = _mean_gap_denominator if method is DenominatorMethod.MEAN_GAP else _regression_denominator

    for season in sorted(by_season):
        teams = by_season[season]
        num_teams = len(teams)
        if num_teams < 2:
            continue

        for cat in categories:
            values = [t.stat_values[cat.key] for t in teams if cat.key in t.stat_values]
            if len(values) < 2:
                continue

            values.sort()
            denominator = compute(values)

            if cat.direction is Direction.LOWER:
                denominator = -denominator

            per_season.append(
                SgpSeasonDenominator(
                    category=cat.key,
                    season=season,
                    denominator=denominator,
                    num_teams=num_teams,
                )
            )

    cat_denoms: dict[str, list[float]] = defaultdict(list)
    for sd in per_season:
        cat_denoms[sd.category].append(sd.denominator)

    averages = {cat: mean(vals) for cat, vals in cat_denoms.items()}

    return SgpDenominators(per_season=tuple(per_season), averages=averages)


def compute_representative_team_totals(
    standings: list[TeamSeasonStats],
    categories: Sequence[CategoryConfig],
) -> dict[str, tuple[float, float]]:
    """Compute average team rate and volume for each rate-stat category.

    For each RATE category, collects (rate, volume) pairs from all teams
    where both ``cat.key`` and ``cat.denominator`` exist in standings.
    Returns ``{cat.key: (mean_rate, mean_volume)}``.
    """
    result: dict[str, tuple[float, float]] = {}
    for cat in categories:
        if cat.stat_type is not StatType.RATE or not cat.denominator:
            continue
        rates: list[float] = []
        volumes: list[float] = []
        for team in standings:
            rate_val = team.stat_values.get(cat.key)
            vol_val = team.stat_values.get(cat.denominator)
            if rate_val is not None and vol_val is not None:
                rates.append(rate_val)
                volumes.append(vol_val)
        if rates:
            result[cat.key] = (mean(rates), mean(volumes))
    return result
