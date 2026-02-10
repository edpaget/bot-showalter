"""Shared feature store for caching per-year player data lookups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.batted_ball_data import (
        PitcherBattedBallDataSource,
        PitcherBattedBallStats,
    )
    from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats, PitcherSkillStats, SkillDataSource
    from fantasy_baseball_manager.pipeline.statcast_data import (
        FullStatcastDataSource,
        StatcastBatterStats,
        StatcastPitcherStats,
    )


@dataclass
class FeatureStore:
    """Lazy, per-year cached lookups for shared pipeline data.

    Loads from the underlying data sources on first access for a given year,
    then returns the cached dict on subsequent calls. Multi-year caching
    supports stages that access different years (e.g. year-1 and year-2).
    """

    statcast_source: FullStatcastDataSource
    batted_ball_source: PitcherBattedBallDataSource
    skill_data_source: SkillDataSource

    _batter_statcast: dict[int, dict[str, StatcastBatterStats]] = field(
        default_factory=dict, init=False, repr=False
    )
    _pitcher_statcast: dict[int, dict[str, StatcastPitcherStats]] = field(
        default_factory=dict, init=False, repr=False
    )
    _pitcher_batted_ball: dict[int, dict[str, PitcherBattedBallStats]] = field(
        default_factory=dict, init=False, repr=False
    )
    _batter_skill: dict[int, dict[str, BatterSkillStats]] = field(
        default_factory=dict, init=False, repr=False
    )
    _pitcher_skill: dict[int, dict[str, PitcherSkillStats]] = field(
        default_factory=dict, init=False, repr=False
    )

    def batter_statcast(self, year: int) -> dict[str, StatcastBatterStats]:
        """Return batter Statcast data keyed by MLBAM ID, loading on first access."""
        if year not in self._batter_statcast:
            data = self.statcast_source.batter_expected_stats(year)
            self._batter_statcast[year] = {s.player_id: s for s in data}
        return self._batter_statcast[year]

    def pitcher_statcast(self, year: int) -> dict[str, StatcastPitcherStats]:
        """Return pitcher Statcast data keyed by MLBAM ID, loading on first access."""
        if year not in self._pitcher_statcast:
            data = self.statcast_source.pitcher_expected_stats(year)
            self._pitcher_statcast[year] = {s.player_id: s for s in data}
        return self._pitcher_statcast[year]

    def pitcher_batted_ball(self, year: int) -> dict[str, PitcherBattedBallStats]:
        """Return pitcher batted ball data keyed by FanGraphs ID, loading on first access."""
        if year not in self._pitcher_batted_ball:
            data = self.batted_ball_source.pitcher_batted_ball_stats(year)
            self._pitcher_batted_ball[year] = {s.player_id: s for s in data}
        return self._pitcher_batted_ball[year]

    def batter_skill(self, year: int) -> dict[str, BatterSkillStats]:
        """Return batter skill data keyed by FanGraphs ID, loading on first access."""
        if year not in self._batter_skill:
            data = self.skill_data_source.batter_skill_stats(year)
            self._batter_skill[year] = {s.player_id: s for s in data}
        return self._batter_skill[year]

    def pitcher_skill(self, year: int) -> dict[str, PitcherSkillStats]:
        """Return pitcher skill data keyed by FanGraphs ID, loading on first access."""
        if year not in self._pitcher_skill:
            data = self.skill_data_source.pitcher_skill_stats(year)
            self._pitcher_skill[year] = {s.player_id: s for s in data}
        return self._pitcher_skill[year]
