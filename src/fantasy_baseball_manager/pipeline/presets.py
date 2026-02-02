from collections.abc import Callable

from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.stages.adjusters import (
    MarcelAgingAdjuster,
    RebaselineAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.stages.playing_time import MarcelPlayingTime
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer


def marcel_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel",
        rate_computer=MarcelRateComputer(),
        adjusters=(RebaselineAdjuster(), MarcelAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


PIPELINES: dict[str, Callable[[], ProjectionPipeline]] = {
    "marcel": marcel_pipeline,
}
