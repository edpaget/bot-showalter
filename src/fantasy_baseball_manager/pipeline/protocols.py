from typing import Protocol

from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.types import PlayerRates


class RateComputer(Protocol):
    def compute_batting_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]: ...

    def compute_pitching_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]: ...


class RateAdjuster(Protocol):
    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]: ...


class PlayingTimeProjector(Protocol):
    def project(self, players: list[PlayerRates]) -> list[PlayerRates]: ...


class ProjectionFinalizer(Protocol):
    def finalize_batting(self, players: list[PlayerRates]) -> list[BattingProjection]: ...

    def finalize_pitching(self, players: list[PlayerRates]) -> list[PitchingProjection]: ...


class ProjectionPipelineProtocol(Protocol):
    def project_batters(self, data_source: StatsDataSource, year: int) -> list[BattingProjection]: ...

    def project_pitchers(self, data_source: StatsDataSource, year: int) -> list[PitchingProjection]: ...
