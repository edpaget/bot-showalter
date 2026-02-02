from dataclasses import dataclass

from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
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
        data_source: StatsDataSource,
        year: int,
    ) -> list[BattingProjection]:
        players = self.rate_computer.compute_batting_rates(data_source, year, self.years_back)
        for adjuster in self.adjusters:
            players = adjuster.adjust(players)
        players = self.playing_time.project(players)
        return self.finalizer.finalize_batting(players)

    def project_pitchers(
        self,
        data_source: StatsDataSource,
        year: int,
    ) -> list[PitchingProjection]:
        players = self.rate_computer.compute_pitching_rates(data_source, year, self.years_back)
        for adjuster in self.adjusters:
            players = adjuster.adjust(players)
        players = self.playing_time.project(players)
        return self.finalizer.finalize_pitching(players)
