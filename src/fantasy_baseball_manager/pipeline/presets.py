from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from fantasy_baseball_manager.cache.factory import create_cache_store
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.park_factors import (
    CachedParkFactorProvider,
    FanGraphsParkFactorProvider,
)
from fantasy_baseball_manager.pipeline.stages.adjusters import (
    MarcelAgingAdjuster,
    RebaselineAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.batter_babip_adjuster import (
    BatterBabipAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.stages.park_factor_adjuster import (
    ParkFactorAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_statcast_adjuster import (
    PitcherStatcastAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.playing_time import MarcelPlayingTime
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)
from fantasy_baseball_manager.pipeline.stages.statcast_adjuster import (
    StatcastRateAdjuster,
)
from fantasy_baseball_manager.pipeline.statcast_data import (
    CachedStatcastDataSource,
    PitcherStatcastDataSource,
    PybaseballStatcastDataSource,
    StatcastDataSource,
)
from fantasy_baseball_manager.player_id.mapper import (
    PlayerIdMapper,
    build_cached_sfbb_mapper,
)


def _cached_statcast_source() -> CachedStatcastDataSource:
    return CachedStatcastDataSource(
        delegate=PybaseballStatcastDataSource(),
        cache=create_cache_store(),
    )


def _default_id_mapper() -> PlayerIdMapper:
    return build_cached_sfbb_mapper(
        cache=create_cache_store(),
        cache_key="presets",
        ttl=7 * 86400,
    )


def _statcast_adjuster(
    statcast_source: StatcastDataSource | None = None,
    id_mapper: PlayerIdMapper | None = None,
) -> StatcastRateAdjuster:
    return StatcastRateAdjuster(
        statcast_source=statcast_source or _cached_statcast_source(),
        id_mapper=id_mapper or _default_id_mapper(),
    )


def _cached_park_factor_provider() -> CachedParkFactorProvider:
    return CachedParkFactorProvider(
        delegate=FanGraphsParkFactorProvider(),
        cache=create_cache_store(),
    )


def marcel_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel",
        rate_computer=MarcelRateComputer(),
        adjusters=(RebaselineAdjuster(), ComponentAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_park_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_park",
        rate_computer=MarcelRateComputer(),
        adjusters=(
            ParkFactorAdjuster(_cached_park_factor_provider()),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_statreg_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    cfg = config or RegressionConfig()
    return ProjectionPipeline(
        name="marcel_statreg",
        rate_computer=StatSpecificRegressionRateComputer(
            batting_regression=cfg.batting_regression_pa,
            pitching_regression=cfg.pitching_regression_outs,
        ),
        adjusters=(RebaselineAdjuster(), ComponentAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_plus_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    cfg = config or RegressionConfig()
    return ProjectionPipeline(
        name="marcel_plus",
        rate_computer=StatSpecificRegressionRateComputer(
            batting_regression=cfg.batting_regression_pa,
            pitching_regression=cfg.pitching_regression_outs,
        ),
        adjusters=(
            ParkFactorAdjuster(_cached_park_factor_provider()),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_norm_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    cfg = config or RegressionConfig()
    return ProjectionPipeline(
        name="marcel_norm",
        rate_computer=StatSpecificRegressionRateComputer(
            batting_regression=cfg.batting_regression_pa,
            pitching_regression=cfg.pitching_regression_outs,
        ),
        adjusters=(
            PitcherNormalizationAdjuster(cfg.pitcher_normalization),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_full_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    cfg = config or RegressionConfig()
    return ProjectionPipeline(
        name="marcel_full",
        rate_computer=StatSpecificRegressionRateComputer(
            batting_regression=cfg.batting_regression_pa,
            pitching_regression=cfg.pitching_regression_outs,
        ),
        adjusters=(
            ParkFactorAdjuster(_cached_park_factor_provider()),
            PitcherNormalizationAdjuster(cfg.pitcher_normalization),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_classic_pipeline() -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_classic",
        rate_computer=MarcelRateComputer(),
        adjusters=(RebaselineAdjuster(), MarcelAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_statcast_pipeline(
    statcast_source: StatcastDataSource | None = None,
    id_mapper: PlayerIdMapper | None = None,
) -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_statcast",
        rate_computer=MarcelRateComputer(),
        adjusters=(
            _statcast_adjuster(statcast_source, id_mapper),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_plus_statcast_pipeline(
    statcast_source: StatcastDataSource | None = None,
    id_mapper: PlayerIdMapper | None = None,
) -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_plus_statcast",
        rate_computer=StatSpecificRegressionRateComputer(),
        adjusters=(
            ParkFactorAdjuster(_cached_park_factor_provider()),
            _statcast_adjuster(statcast_source, id_mapper),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_full_statcast_pipeline(
    statcast_source: StatcastDataSource | None = None,
    id_mapper: PlayerIdMapper | None = None,
) -> ProjectionPipeline:
    return ProjectionPipeline(
        name="marcel_full_statcast",
        rate_computer=StatSpecificRegressionRateComputer(),
        adjusters=(
            ParkFactorAdjuster(_cached_park_factor_provider()),
            PitcherNormalizationAdjuster(),
            _statcast_adjuster(statcast_source, id_mapper),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_full_statcast_babip_pipeline(
    statcast_source: StatcastDataSource | None = None,
    id_mapper: PlayerIdMapper | None = None,
) -> ProjectionPipeline:
    source = statcast_source or _cached_statcast_source()
    mapper = id_mapper or _default_id_mapper()
    return ProjectionPipeline(
        name="marcel_full_statcast_babip",
        rate_computer=StatSpecificRegressionRateComputer(),
        adjusters=(
            ParkFactorAdjuster(_cached_park_factor_provider()),
            PitcherNormalizationAdjuster(),
            StatcastRateAdjuster(statcast_source=source, id_mapper=mapper),
            BatterBabipAdjuster(statcast_source=source, id_mapper=mapper),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_full_statcast_pitching_pipeline(
    statcast_source: StatcastDataSource | None = None,
    pitcher_statcast_source: PitcherStatcastDataSource | None = None,
    id_mapper: PlayerIdMapper | None = None,
) -> ProjectionPipeline:
    source = statcast_source or _cached_statcast_source()
    mapper = id_mapper or _default_id_mapper()
    p_source = cast(PitcherStatcastDataSource, pitcher_statcast_source or source)
    return ProjectionPipeline(
        name="marcel_full_statcast_pitching",
        rate_computer=StatSpecificRegressionRateComputer(),
        adjusters=(
            ParkFactorAdjuster(_cached_park_factor_provider()),
            PitcherNormalizationAdjuster(),
            PitcherStatcastAdjuster(statcast_source=p_source, id_mapper=mapper),
            StatcastRateAdjuster(statcast_source=source, id_mapper=mapper),
            BatterBabipAdjuster(statcast_source=source, id_mapper=mapper),
            RebaselineAdjuster(),
            ComponentAgingAdjuster(),
        ),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


PIPELINES: dict[str, Callable[[], ProjectionPipeline]] = {
    "marcel_classic": marcel_classic_pipeline,
    "marcel": marcel_pipeline,
    "marcel_park": marcel_park_pipeline,
    "marcel_statreg": marcel_statreg_pipeline,
    "marcel_plus": marcel_plus_pipeline,
    "marcel_norm": marcel_norm_pipeline,
    "marcel_full": marcel_full_pipeline,
    "marcel_statcast": marcel_statcast_pipeline,
    "marcel_plus_statcast": marcel_plus_statcast_pipeline,
    "marcel_full_statcast": marcel_full_statcast_pipeline,
    "marcel_full_statcast_babip": marcel_full_statcast_babip_pipeline,
    "marcel_full_statcast_pitching": marcel_full_statcast_pitching_pipeline,
}

_CONFIGURABLE_FACTORIES: dict[str, Callable[[RegressionConfig | None], ProjectionPipeline]] = {
    "marcel_statreg": marcel_statreg_pipeline,
    "marcel_plus": marcel_plus_pipeline,
    "marcel_norm": marcel_norm_pipeline,
    "marcel_full": marcel_full_pipeline,
}


def build_pipeline(
    name: str,
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Build a pipeline by name, optionally passing a RegressionConfig.

    Pipelines that accept config (marcel_statreg, marcel_plus, marcel_norm,
    marcel_full) will receive the given config.  Others ignore it.
    """
    if name in _CONFIGURABLE_FACTORIES:
        return _CONFIGURABLE_FACTORIES[name](config)
    if name in PIPELINES:
        return PIPELINES[name]()
    raise ValueError(f"Unknown pipeline: {name}")
