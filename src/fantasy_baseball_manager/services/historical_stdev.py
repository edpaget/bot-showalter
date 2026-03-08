"""Compute per-category standard deviations from historical actual stats."""

import statistics
from typing import TYPE_CHECKING

from fantasy_baseball_manager.models.zar.engine import convert_rate_stats
from fantasy_baseball_manager.services.stats_conversion import stats_to_dict

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings
    from fantasy_baseball_manager.repos import BattingStatsRepo, PitchingStatsRepo


def compute_historical_stdevs(
    seasons: list[int],
    league: LeagueSettings,
    batting_repo: BattingStatsRepo,
    pitching_repo: PitchingStatsRepo,
    *,
    actuals_source: str = "fangraphs",
) -> dict[str, float]:
    """Compute per-category population stdevs from historical actuals, averaged across seasons.

    For each season, loads actual stats, converts rate stats to marginal contributions
    (same transform the ZAR engine applies), then computes pstdev per category. Returns
    the average stdev across all seasons that had data for each category.
    """
    batting_categories = list(league.batting_categories)
    pitching_categories = list(league.pitching_categories)
    batting_keys = [cat.key for cat in batting_categories]
    pitching_keys = [cat.key for cat in pitching_categories]

    # Collect per-season stdevs for each category
    season_stdevs: dict[str, list[float]] = {k: [] for k in batting_keys + pitching_keys}

    for season in seasons:
        # Batting pool
        if batting_categories:
            batting_actuals = batting_repo.get_by_season(season, source=actuals_source)
            batter_dicts = [stats_to_dict(bs) for bs in batting_actuals if (bs.pa or 0) > 0]
            if batter_dicts:
                converted = convert_rate_stats(batter_dicts, batting_categories)
                for key in batting_keys:
                    values = [row.get(key, 0.0) for row in converted]
                    season_stdevs[key].append(statistics.pstdev(values))

        # Pitching pool
        if pitching_categories:
            pitching_actuals = pitching_repo.get_by_season(season, source=actuals_source)
            pitcher_dicts = [stats_to_dict(ps) for ps in pitching_actuals if (ps.ip or 0) > 0]
            if pitcher_dicts:
                converted = convert_rate_stats(pitcher_dicts, pitching_categories)
                for key in pitching_keys:
                    values = [row.get(key, 0.0) for row in converted]
                    season_stdevs[key].append(statistics.pstdev(values))

    # Average across seasons (skip categories with no data)
    result: dict[str, float] = {}
    for key, stdevs in season_stdevs.items():
        if stdevs:
            result[key] = sum(stdevs) / len(stdevs)
        else:
            result[key] = 0.0

    return result
