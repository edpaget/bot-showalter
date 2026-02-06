from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import (
        BattingProjection,
        BattingSeasonStats,
        PitchingProjection,
        PitchingSeasonStats,
    )
    from fantasy_baseball_manager.pipeline.protocols import ProjectionPipelineProtocol


class PipelineProjectionSource:
    """Wraps a pipeline + DataSources + year into the ProjectionSource protocol."""

    def __init__(
        self,
        pipeline: ProjectionPipelineProtocol,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        year: int,
    ) -> None:
        self._pipeline = pipeline
        self._batting_source = batting_source
        self._team_batting_source = team_batting_source
        self._pitching_source = pitching_source
        self._team_pitching_source = team_pitching_source
        self._year = year

    def batting_projections(self) -> list[BattingProjection]:
        return self._pipeline.project_batters(self._batting_source, self._team_batting_source, self._year)

    def pitching_projections(self) -> list[PitchingProjection]:
        return self._pipeline.project_pitchers(self._pitching_source, self._team_pitching_source, self._year)
