from collections.abc import Callable

from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.park_factors import FanGraphsParkFactorProvider
from fantasy_baseball_manager.pipeline.stages.adjusters import (
    MarcelAgingAdjuster,
    RebaselineAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.stages.park_factor_adjuster import (
    ParkFactorAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.playing_time import MarcelPlayingTime
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)


def marcel_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel",
        rate_computer=MarcelRateComputer(),
        adjusters=(RebaselineAdjuster(), MarcelAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_park_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_park",
        rate_computer=MarcelRateComputer(),
        adjusters=(
            ParkFactorAdjuster(FanGraphsParkFactorProvider()),
            RebaselineAdjuster(),
            MarcelAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_statreg_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_statreg",
        rate_computer=StatSpecificRegressionRateComputer(),
        adjusters=(RebaselineAdjuster(), MarcelAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_plus_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_plus",
        rate_computer=StatSpecificRegressionRateComputer(),
        adjusters=(
            ParkFactorAdjuster(FanGraphsParkFactorProvider()),
            RebaselineAdjuster(),
            MarcelAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


PIPELINES: dict[str, Callable[[], ProjectionPipeline]] = {
    "marcel": marcel_pipeline,
    "marcel_park": marcel_park_pipeline,
    "marcel_statreg": marcel_statreg_pipeline,
    "marcel_plus": marcel_plus_pipeline,
}
