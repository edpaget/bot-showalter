from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline


class PipelineProjectionSource:
    """Wraps a pipeline + data_source + year into the ProjectionSource protocol."""

    def __init__(
        self,
        pipeline: ProjectionPipeline,
        data_source: StatsDataSource,
        year: int,
    ) -> None:
        self._pipeline = pipeline
        self._data_source = data_source
        self._year = year

    def batting_projections(self) -> list[BattingProjection]:
        return self._pipeline.project_batters(self._data_source, self._year)

    def pitching_projections(self) -> list[PitchingProjection]:
        return self._pipeline.project_pitchers(self._data_source, self._year)
