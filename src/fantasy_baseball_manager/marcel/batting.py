from fantasy_baseball_manager.marcel.age_adjustment import age_multiplier
from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.marcel.league_averages import (
    BATTING_COMPONENT_STATS,
    compute_batting_league_rates,
    rebaseline,
)
from fantasy_baseball_manager.marcel.models import BattingProjection, BattingSeasonStats
from fantasy_baseball_manager.marcel.weights import projected_pa, weighted_rate

YEAR_WEIGHTS = [5, 4, 3]
REGRESSION_PA = 1200


def project_batters(
    data_source: StatsDataSource,
    year: int,
) -> list[BattingProjection]:
    """Generate MARCEL batting projections for the given year.

    Uses 3 prior years of data from data_source.
    """
    years = [year - 1, year - 2, year - 3]

    # Fetch player stats for each prior year
    player_seasons: dict[int, list[BattingSeasonStats]] = {}
    for y in years:
        player_seasons[y] = data_source.batting_stats(y)

    # Fetch league totals for each prior year
    league_rates: dict[int, dict[str, float]] = {}
    for y in years:
        team_stats = data_source.team_batting(y)
        if team_stats:
            league_rates[y] = compute_batting_league_rates(team_stats)

    # Use most recent year's league rates as the target
    target_rates = league_rates[years[0]]

    # Average league rates across all available years for regression
    avg_league_rates: dict[str, float] = {}
    for stat in BATTING_COMPONENT_STATS:
        rates_for_stat = [league_rates[y][stat] for y in years if y in league_rates]
        avg_league_rates[stat] = sum(rates_for_stat) / len(rates_for_stat)

    # Group player data by player_id across years
    player_data: dict[str, dict[int, BattingSeasonStats]] = {}
    for y in years:
        for p in player_seasons.get(y, []):
            if p.player_id not in player_data:
                player_data[p.player_id] = {}
            player_data[p.player_id][y] = p

    projections: list[BattingProjection] = []
    for player_id, seasons in player_data.items():
        # Use most recent available season for name/age
        most_recent = next(seasons[y] for y in years if y in seasons)
        projection_age = most_recent.age + (year - most_recent.year)

        # Collect PA per year (0 for missing years)
        pa_per_year = [seasons[y].pa if y in seasons else 0 for y in years]

        # Compute weighted rates for each component stat
        raw_rates: dict[str, float] = {}
        for stat in BATTING_COMPONENT_STATS:
            stat_per_year = [getattr(seasons[y], stat) if y in seasons else 0 for y in years]
            weights = YEAR_WEIGHTS[: len(years)]
            raw_rates[stat] = weighted_rate(
                stats=stat_per_year,
                opportunities=pa_per_year,
                weights=weights,
                league_rate=avg_league_rates[stat],
                regression_pa=REGRESSION_PA,
            )

        # Rebaseline to most recent year's league environment
        rates = rebaseline(raw_rates, target_rates)

        # Age adjustment
        mult = age_multiplier(projection_age)
        rates = {stat: rate * mult for stat, rate in rates.items()}

        # Project playing time
        proj_pa = projected_pa(
            pa_y1=pa_per_year[0],
            pa_y2=pa_per_year[1] if len(pa_per_year) > 1 else 0,
        )

        # Compute counting stats
        projected_stats = {stat: rate * proj_pa for stat, rate in rates.items()}

        # Derive AB and H from component stats
        proj_h = (
            projected_stats["singles"] + projected_stats["doubles"] + projected_stats["triples"] + projected_stats["hr"]
        )
        proj_ab = (
            proj_pa - projected_stats["bb"] - projected_stats["hbp"] - projected_stats["sf"] - projected_stats["sh"]
        )

        projections.append(
            BattingProjection(
                player_id=player_id,
                name=most_recent.name,
                year=year,
                age=projection_age,
                pa=proj_pa,
                ab=proj_ab,
                h=proj_h,
                singles=projected_stats["singles"],
                doubles=projected_stats["doubles"],
                triples=projected_stats["triples"],
                hr=projected_stats["hr"],
                bb=projected_stats["bb"],
                so=projected_stats["so"],
                hbp=projected_stats["hbp"],
                sf=projected_stats["sf"],
                sh=projected_stats["sh"],
                sb=projected_stats["sb"],
                cs=projected_stats["cs"],
            )
        )

    return projections
