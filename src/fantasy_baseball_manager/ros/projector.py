from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.ros.protocol import ProjectionBlender


class ROSProjector:
    def __init__(
        self,
        pipeline: ProjectionPipeline,
        data_source: StatsDataSource,
        blender: ProjectionBlender,
    ) -> None:
        self._pipeline = pipeline
        self._data_source = data_source
        self._blender = blender

    def project_batters(self, year: int) -> list[BattingProjection]:
        preseason = self._pipeline.project_batters(self._data_source, year)
        actuals_list = self._data_source.batting_stats(year)
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
        preseason = self._pipeline.project_pitchers(self._data_source, year)
        actuals_list = self._data_source.pitching_stats(year)
        actuals_by_id = {a.player_id: a for a in actuals_list}

        results: list[PitchingProjection] = []
        for proj in preseason:
            actuals = actuals_by_id.get(proj.player_id)
            if actuals is not None:
                results.append(self._blender.blend_pitching(proj, actuals))
            else:
                results.append(proj)
        return results
