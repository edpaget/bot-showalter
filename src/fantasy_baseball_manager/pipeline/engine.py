from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.context import get_context, new_context
from fantasy_baseball_manager.result import Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource, DataSourceError
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
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
    )
    from fantasy_baseball_manager.result import Err


def _adapt_batting(data_source: StatsDataSource) -> DataSource[BattingSeasonStats]:
    """Wrap StatsDataSource.batting_stats as a DataSource[BattingSeasonStats]."""

    def _source(_query: Any) -> Ok[list[BattingSeasonStats]] | Err[DataSourceError]:
        year = get_context().year
        return Ok(data_source.batting_stats(year))

    return _source  # type: ignore[return-value]


def _adapt_team_batting(data_source: StatsDataSource) -> DataSource[BattingSeasonStats]:
    """Wrap StatsDataSource.team_batting as a DataSource[BattingSeasonStats]."""

    def _source(_query: Any) -> Ok[list[BattingSeasonStats]] | Err[DataSourceError]:
        year = get_context().year
        return Ok(data_source.team_batting(year))

    return _source  # type: ignore[return-value]


def _adapt_pitching(data_source: StatsDataSource) -> DataSource[PitchingSeasonStats]:
    """Wrap StatsDataSource.pitching_stats as a DataSource[PitchingSeasonStats]."""

    def _source(_query: Any) -> Ok[list[PitchingSeasonStats]] | Err[DataSourceError]:
        year = get_context().year
        return Ok(data_source.pitching_stats(year))

    return _source  # type: ignore[return-value]


def _adapt_team_pitching(data_source: StatsDataSource) -> DataSource[PitchingSeasonStats]:
    """Wrap StatsDataSource.team_pitching as a DataSource[PitchingSeasonStats]."""

    def _source(_query: Any) -> Ok[list[PitchingSeasonStats]] | Err[DataSourceError]:
        year = get_context().year
        return Ok(data_source.team_pitching(year))

    return _source  # type: ignore[return-value]


def _uses_datasource_protocol(rate_computer: Any) -> bool:
    """Check if rate_computer uses new DataSource-based signature.

    Detects the calling convention by inspecting parameter names.
    New-style: compute_batting_rates(batting_source, team_batting_source, year, years_back)
    Legacy:    compute_batting_rates(data_source, year, years_back)
    """
    try:
        sig = inspect.signature(rate_computer.compute_batting_rates)
        return "batting_source" in sig.parameters
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class ProjectionPipeline:
    name: str
    # TODO: Restore RateComputer type after migrating all rate computer implementations
    rate_computer: Any
    adjusters: tuple[RateAdjuster, ...]
    playing_time: PlayingTimeProjector
    finalizer: ProjectionFinalizer
    years_back: int = 3

    def project_batters(
        self,
        data_source: StatsDataSource,
        year: int,
    ) -> list[BattingProjection]:
        if _uses_datasource_protocol(self.rate_computer):
            batting_source = _adapt_batting(data_source)
            team_batting_source = _adapt_team_batting(data_source)
            with new_context(year=year):
                players = self.rate_computer.compute_batting_rates(
                    batting_source, team_batting_source, year, self.years_back
                )
        else:
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
        if _uses_datasource_protocol(self.rate_computer):
            pitching_source = _adapt_pitching(data_source)
            team_pitching_source = _adapt_team_pitching(data_source)
            with new_context(year=year):
                players = self.rate_computer.compute_pitching_rates(
                    pitching_source, team_pitching_source, year, self.years_back
                )
        else:
            players = self.rate_computer.compute_pitching_rates(data_source, year, self.years_back)
        for adjuster in self.adjusters:
            players = adjuster.adjust(players)
        players = self.playing_time.project(players)
        return self.finalizer.finalize_pitching(players)
