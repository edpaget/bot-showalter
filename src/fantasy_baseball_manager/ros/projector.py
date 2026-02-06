from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.context import new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import (
        BattingProjection,
        BattingSeasonStats,
        PitchingProjection,
        PitchingSeasonStats,
    )
    from fantasy_baseball_manager.pipeline.protocols import ProjectionPipelineProtocol
    from fantasy_baseball_manager.ros.protocol import ProjectionBlender


class ROSProjector:
    def __init__(
        self,
        pipeline: ProjectionPipelineProtocol,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        blender: ProjectionBlender,
    ) -> None:
        self._pipeline = pipeline
        self._batting_source = batting_source
        self._team_batting_source = team_batting_source
        self._pitching_source = pitching_source
        self._team_pitching_source = team_pitching_source
        self._blender = blender

    def project_batters(self, year: int) -> list[BattingProjection]:
        preseason = self._pipeline.project_batters(self._batting_source, self._team_batting_source, year)
        with new_context(year=year):
            result = self._batting_source(ALL_PLAYERS)
        actuals_list = list(result.unwrap()) if result.is_ok() else []
        actuals_by_id = {a.player_id: a for a in actuals_list}

        results: list[BattingProjection] = []
        for proj in preseason:
            actuals = actuals_by_id.get(proj.player_id)
            if actuals is not None:
                results.append(self._blender.blend_batting(proj, actuals))
            else:
                results.append(proj)
        return results

    def project_pitchers(self, year: int) -> list[PitchingProjection]:
        preseason = self._pipeline.project_pitchers(self._pitching_source, self._team_pitching_source, year)
        with new_context(year=year):
            result = self._pitching_source(ALL_PLAYERS)
        actuals_list = list(result.unwrap()) if result.is_ok() else []
        actuals_by_id = {a.player_id: a for a in actuals_list}

        results: list[PitchingProjection] = []
        for proj in preseason:
            actuals = actuals_by_id.get(proj.player_id)
            if actuals is not None:
                results.append(self._blender.blend_pitching(proj, actuals))
            else:
                results.append(proj)
        return results
