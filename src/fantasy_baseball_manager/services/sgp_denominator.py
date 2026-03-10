from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    Direction,
    SgpDenominators,
    SgpSeasonDenominator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.domain import CategoryConfig, TeamSeasonStats


def compute_sgp_denominators(
    standings: list[TeamSeasonStats],
    categories: Sequence[CategoryConfig],
) -> SgpDenominators:
    """Compute SGP denominators from league standings data.

    For each category and season, sorts teams by stat value, computes the mean
    gap between adjacent teams, and returns that as the denominator. For
    "lower is better" categories (ERA, WHIP), the denominator is negated so
    it is always positive.
    """
    by_season: dict[int, list[TeamSeasonStats]] = defaultdict(list)
    for team in standings:
        by_season[team.season].append(team)

    per_season: list[SgpSeasonDenominator] = []

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
            gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            denominator = mean(gaps)

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
