from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.marcel.league_averages import (
    BATTING_COMPONENT_STATS,
    PITCHING_COMPONENT_STATS,
    compute_batting_league_rates,
    compute_pitching_league_rates,
)
from fantasy_baseball_manager.marcel.weights import weighted_rate
from fantasy_baseball_manager.pipeline.stages.rate_computers import (
    MARCEL_BATTING_WEIGHTS,
    MARCEL_PITCHING_WEIGHTS,
    MARCEL_REGRESSION_OUTS,
    MARCEL_REGRESSION_PA,
    STARTER_GS_RATIO,
)
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats


class StatSpecificRegressionRateComputer:
    """Marcel rate computer with per-stat regression constants.

    Identical to MarcelRateComputer except each stat uses its own
    regression amount (PA for batting, outs for pitching) instead
    of a single constant for all stats.
    """

    def __init__(
        self,
        batting_regression: dict[str, float] | None = None,
        pitching_regression: dict[str, float] | None = None,
    ) -> None:
        self._batting_regression = batting_regression or BATTING_REGRESSION_PA
        self._pitching_regression = pitching_regression or PITCHING_REGRESSION_OUTS

    def compute_batting_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        years = [year - i for i in range(1, years_back + 1)]
        weights = list(MARCEL_BATTING_WEIGHTS[:years_back])

        player_seasons: dict[int, list[BattingSeasonStats]] = {}
        league_rates: dict[int, dict[str, float]] = {}
        for y in years:
            player_seasons[y] = data_source.batting_stats(y)
            team_stats = data_source.team_batting(y)
            if team_stats:
                league_rates[y] = compute_batting_league_rates(team_stats)

        target_rates = league_rates[years[0]]

        avg_league_rates: dict[str, float] = {}
        for stat in BATTING_COMPONENT_STATS:
            rates_for_stat = [league_rates[y][stat] for y in years if y in league_rates]
            avg_league_rates[stat] = sum(rates_for_stat) / len(rates_for_stat)

        player_data: dict[str, dict[int, BattingSeasonStats]] = {}
        for y in years:
            for p in player_seasons.get(y, []):
                if p.player_id not in player_data:
                    player_data[p.player_id] = {}
                player_data[p.player_id][y] = p

        result: list[PlayerRates] = []
        for player_id, seasons in player_data.items():
            most_recent = next(seasons[y] for y in years if y in seasons)
            projection_age = most_recent.age + (year - most_recent.year)

            pa_per_year: list[float] = [float(seasons[y].pa) if y in seasons else 0.0 for y in years]

            raw_rates: dict[str, float] = {}
            for stat in BATTING_COMPONENT_STATS:
                stat_per_year: list[float] = [float(getattr(seasons[y], stat)) if y in seasons else 0.0 for y in years]
                regression_pa = self._batting_regression.get(stat, MARCEL_REGRESSION_PA)
                raw_rates[stat] = weighted_rate(
                    stats=stat_per_year,
                    opportunities=pa_per_year,
                    weights=weights,
                    league_rate=avg_league_rates[stat],
                    regression_pa=regression_pa,
                )

            result.append(
                PlayerRates(
                    player_id=player_id,
                    name=most_recent.name,
                    year=year,
                    age=projection_age,
                    rates=raw_rates,
                    metadata={
                        "pa_per_year": pa_per_year,
                        "avg_league_rates": avg_league_rates,
                        "target_rates": target_rates,
                        "team": most_recent.team,
                    },
                )
            )

        return result

    def compute_pitching_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        years = [year - i for i in range(1, years_back + 1)]
        weights = list(MARCEL_PITCHING_WEIGHTS[:years_back])

        player_seasons: dict[int, list[PitchingSeasonStats]] = {}
        league_rates: dict[int, dict[str, float]] = {}
        for y in years:
            player_seasons[y] = data_source.pitching_stats(y)
            team_stats = data_source.team_pitching(y)
            if team_stats:
                league_rates[y] = compute_pitching_league_rates(team_stats)

        target_rates = league_rates[years[0]]

        avg_league_rates: dict[str, float] = {}
        for stat in PITCHING_COMPONENT_STATS:
            rates_for_stat = [league_rates[y][stat] for y in years if y in league_rates]
            avg_league_rates[stat] = sum(rates_for_stat) / len(rates_for_stat)

        player_data: dict[str, dict[int, PitchingSeasonStats]] = {}
        for y in years:
            for p in player_seasons.get(y, []):
                if p.player_id not in player_data:
                    player_data[p.player_id] = {}
                player_data[p.player_id][y] = p

        result: list[PlayerRates] = []
        for player_id, seasons in player_data.items():
            most_recent = next(seasons[y] for y in years if y in seasons)
            projection_age = most_recent.age + (year - most_recent.year)

            outs_per_year: list[float] = [seasons[y].ip * 3 if y in seasons else 0.0 for y in years]
            ip_per_year = [seasons[y].ip if y in seasons else 0.0 for y in years]

            total_gs = sum(s.gs for s in seasons.values())
            total_g = sum(s.g for s in seasons.values())
            is_starter = (total_gs / total_g) >= STARTER_GS_RATIO if total_g > 0 else True

            raw_rates: dict[str, float] = {}
            for stat in PITCHING_COMPONENT_STATS:
                stat_per_year: list[float] = [float(getattr(seasons[y], stat)) if y in seasons else 0.0 for y in years]
                regression_outs = self._pitching_regression.get(stat, MARCEL_REGRESSION_OUTS)
                raw_rates[stat] = weighted_rate(
                    stats=stat_per_year,
                    opportunities=outs_per_year,
                    weights=weights,
                    league_rate=avg_league_rates[stat],
                    regression_pa=regression_outs,
                )

            result.append(
                PlayerRates(
                    player_id=player_id,
                    name=most_recent.name,
                    year=year,
                    age=projection_age,
                    rates=raw_rates,
                    metadata={
                        "ip_per_year": ip_per_year,
                        "is_starter": is_starter,
                        "avg_league_rates": avg_league_rates,
                        "target_rates": target_rates,
                        "team": most_recent.team,
                    },
                )
            )

        return result
