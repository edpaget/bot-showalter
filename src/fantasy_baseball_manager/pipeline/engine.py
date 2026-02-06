from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.context import new_context

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import (
        BattingProjection,
        BattingSeasonStats,
        PitchingProjection,
        PitchingSeasonStats,
    )
    from fantasy_baseball_manager.pipeline.protocols import (
        PlayingTimeProjector,
        ProjectionFinalizer,
        RateAdjuster,
        RateComputer,
    )


@dataclass(frozen=True)
class ProjectionPipeline:
    name: str
    rate_computer: RateComputer
    adjusters: tuple[RateAdjuster, ...]
    playing_time: PlayingTimeProjector
    finalizer: ProjectionFinalizer
    years_back: int = 3

    def project_batters(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
    ) -> list[BattingProjection]:
        with new_context(year=year):
            players = self.rate_computer.compute_batting_rates(
                batting_source, team_batting_source, year, self.years_back
            )
        for adjuster in self.adjusters:
            players = adjuster.adjust(players)
        players = self.playing_time.project(players)
        return self.finalizer.finalize_batting(players)

    def project_pitchers(
        self,
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        year: int,
    ) -> list[PitchingProjection]:
        with new_context(year=year):
            players = self.rate_computer.compute_pitching_rates(
                pitching_source, team_pitching_source, year, self.years_back
            )
        for adjuster in self.adjusters:
            players = adjuster.adjust(players)
        players = self.playing_time.project(players)
        return self.finalizer.finalize_pitching(players)
