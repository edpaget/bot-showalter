from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.marcel.models import (
        BattingProjection,
        BattingSeasonStats,
        PitchingProjection,
        PitchingSeasonStats,
    )
    from fantasy_baseball_manager.pipeline.types import PlayerRates


class RateComputer(Protocol):
    def compute_batting_rates(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]: ...

    def compute_pitching_rates(
        self,
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
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
