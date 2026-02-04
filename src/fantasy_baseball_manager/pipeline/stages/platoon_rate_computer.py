"""Platoon-aware rate computer that projects batting rates by pitcher handedness.

Computes rates separately for vs-LHP and vs-RHP matchups, applies heavier
regression (2x by default) to account for smaller split samples, then blends
by expected matchup frequency.  Pitching delegates unchanged to a wrapped
RateComputer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.marcel.league_averages import (
    BATTING_COMPONENT_STATS,
    compute_batting_league_rates,
)
from fantasy_baseball_manager.marcel.weights import weighted_rate
from fantasy_baseball_manager.pipeline.stages.rate_computers import (
    MARCEL_BATTING_WEIGHTS,
)
from fantasy_baseball_manager.pipeline.stages.split_regression_constants import (
    BATTING_SPLIT_REGRESSION_PA,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats
    from fantasy_baseball_manager.pipeline.protocols import RateComputer
    from fantasy_baseball_manager.pipeline.stages.split_data_source import (
        SplitStatsDataSource,
    )


class PlatoonRateComputer:
    """Marcel rate computer with platoon split awareness.

    Computes batting rates separately for vs-LHP and vs-RHP, then blends
    by expected matchup frequency.  Pitching delegates to a wrapped
    RateComputer.
    """

    def __init__(
        self,
        split_source: SplitStatsDataSource,
        pitching_delegate: RateComputer,
        batting_regression: dict[str, float] | None = None,
        pct_vs_rhp: float = 0.72,
        pct_vs_lhp: float = 0.28,
    ) -> None:
        self._split_source = split_source
        self._pitching_delegate = pitching_delegate
        self._batting_regression = batting_regression or BATTING_SPLIT_REGRESSION_PA
        self._pct_vs_rhp = pct_vs_rhp
        self._pct_vs_lhp = pct_vs_lhp

    def compute_batting_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        years = [year - i for i in range(1, years_back + 1)]
        weights = list(MARCEL_BATTING_WEIGHTS[:years_back])

        # Fetch split stats for each historical year
        lhp_seasons: dict[int, list[BattingSeasonStats]] = {}
        rhp_seasons: dict[int, list[BattingSeasonStats]] = {}
        league_rates: dict[int, dict[str, float]] = {}

        for y in years:
            lhp_seasons[y] = self._split_source.batting_stats_vs_lhp(y)
            rhp_seasons[y] = self._split_source.batting_stats_vs_rhp(y)
            team_stats = data_source.team_batting(y)
            if team_stats:
                league_rates[y] = compute_batting_league_rates(team_stats)

        target_rates = league_rates[years[0]]

        avg_league_rates: dict[str, float] = {}
        for stat in BATTING_COMPONENT_STATS:
            rates_for_stat = [league_rates[y][stat] for y in years if y in league_rates]
            avg_league_rates[stat] = sum(rates_for_stat) / len(rates_for_stat)

        # Index players by split
        lhp_player_data: dict[str, dict[int, BattingSeasonStats]] = {}
        rhp_player_data: dict[str, dict[int, BattingSeasonStats]] = {}

        for y in years:
            for p in lhp_seasons.get(y, []):
                lhp_player_data.setdefault(p.player_id, {})[y] = p
            for p in rhp_seasons.get(y, []):
                rhp_player_data.setdefault(p.player_id, {})[y] = p

        all_player_ids = set(lhp_player_data.keys()) | set(rhp_player_data.keys())

        result: list[PlayerRates] = []
        for player_id in all_player_ids:
            lhp_data = lhp_player_data.get(player_id, {})
            rhp_data = rhp_player_data.get(player_id, {})

            # Determine player info from most recent season in either split
            most_recent = self._most_recent_player(lhp_data, rhp_data, years)
            projection_age = most_recent.age + (year - most_recent.year)

            # Compute rates for each split
            rates_vs_lhp = self._compute_split_rates(lhp_data, years, weights, avg_league_rates)
            rates_vs_rhp = self._compute_split_rates(rhp_data, years, weights, avg_league_rates)

            # Blend by matchup frequency
            blended_rates: dict[str, float] = {}
            for stat in BATTING_COMPONENT_STATS:
                blended_rates[stat] = self._pct_vs_lhp * rates_vs_lhp[stat] + self._pct_vs_rhp * rates_vs_rhp[stat]

            # PA per year = sum of both splits
            pa_per_year: list[float] = []
            for y in years:
                lhp_pa = float(lhp_data[y].pa) if y in lhp_data else 0.0
                rhp_pa = float(rhp_data[y].pa) if y in rhp_data else 0.0
                pa_per_year.append(lhp_pa + rhp_pa)

            result.append(
                PlayerRates(
                    player_id=player_id,
                    name=most_recent.name,
                    year=year,
                    age=projection_age,
                    rates=blended_rates,
                    metadata={
                        "rates_vs_lhp": rates_vs_lhp,
                        "rates_vs_rhp": rates_vs_rhp,
                        "pct_vs_rhp": self._pct_vs_rhp,
                        "pct_vs_lhp": self._pct_vs_lhp,
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
        return self._pitching_delegate.compute_pitching_rates(data_source, year, years_back)

    def _compute_split_rates(
        self,
        player_data: dict[int, BattingSeasonStats],
        years: list[int],
        weights: Sequence[float],
        avg_league_rates: dict[str, float],
    ) -> dict[str, float]:
        """Compute regressed rates for a single split."""
        pa_per_year: list[float] = [float(player_data[y].pa) if y in player_data else 0.0 for y in years]

        rates: dict[str, float] = {}
        for stat in BATTING_COMPONENT_STATS:
            stat_per_year: list[float] = [
                float(getattr(player_data[y], stat)) if y in player_data else 0.0 for y in years
            ]
            regression_pa = self._batting_regression.get(stat, BATTING_SPLIT_REGRESSION_PA.get(stat, 1200))
            rates[stat] = weighted_rate(
                stats=stat_per_year,
                opportunities=pa_per_year,
                weights=weights,
                league_rate=avg_league_rates[stat],
                regression_pa=regression_pa,
            )
        return rates

    @staticmethod
    def _most_recent_player(
        lhp_data: dict[int, BattingSeasonStats],
        rhp_data: dict[int, BattingSeasonStats],
        years: list[int],
    ) -> BattingSeasonStats:
        """Return the most recent season record for a player from either split."""
        for y in years:
            if y in rhp_data:
                return rhp_data[y]
            if y in lhp_data:
                return lhp_data[y]
        # Should not happen since player is in at least one split
        raise ValueError("Player has no data in any year")
