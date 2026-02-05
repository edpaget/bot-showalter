"""Pre-configured projection pipelines.

Available pipelines:
- marcel_classic: Original Marcel with uniform aging
- marcel: Modern baseline with per-stat regression and component aging
- marcel_full: Kitchen-sink with park factors, Statcast, and BABIP adjustments
- marcel_gb: marcel_full + gradient boosting residual corrections (best accuracy)
- mle: ML-based Minor League Equivalencies for projecting players with limited MLB history
- steamer: Steamer projections from FanGraphs
- zips: ZiPS projections from FanGraphs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from fantasy_baseball_manager.cache.factory import create_cache_store
from fantasy_baseball_manager.cache.sources import CachedProjectionSource
from fantasy_baseball_manager.pipeline.builder import PipelineBuilder
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.stages.adjusters import (
    MarcelAgingAdjuster,
    RebaselineAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import ComponentAgingAdjuster
from fantasy_baseball_manager.pipeline.stages.finalizers import StandardFinalizer
from fantasy_baseball_manager.pipeline.stages.playing_time import MarcelPlayingTime
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)
from fantasy_baseball_manager.projections.adapter import ExternalProjectionAdapter
from fantasy_baseball_manager.projections.fangraphs import FanGraphsProjectionSource
from fantasy_baseball_manager.projections.models import ProjectionSystem


def marcel_classic_pipeline() -> ProjectionPipeline:
    """Original Marcel method: MarcelRateComputer + uniform aging."""
    return ProjectionPipeline(
        name="marcel_classic",
        rate_computer=MarcelRateComputer(),
        adjusters=(RebaselineAdjuster(), MarcelAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Modern baseline: per-stat regression + component aging.

    This is the simplest pipeline, using only historical stats without
    any Statcast or external data adjustments.
    """
    cfg = config or RegressionConfig()
    return ProjectionPipeline(
        name="marcel",
        rate_computer=StatSpecificRegressionRateComputer(
            batting_regression=cfg.batting_regression_pa,
            pitching_regression=cfg.pitching_regression_outs,
        ),
        adjusters=(RebaselineAdjuster(), ComponentAgingAdjuster()),
        playing_time=MarcelPlayingTime(),
        finalizer=StandardFinalizer(),
        years_back=3,
    )


def marcel_full_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Kitchen-sink pipeline with all adjusters enabled.

    Includes: park factors, pitcher normalization, Statcast xStats,
    batter BABIP skill adjustment, rebaseline, and component aging.
    """
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("marcel_full", config=cfg)
        .with_park_factors()
        .with_pitcher_normalization()
        .with_pitcher_statcast()
        .with_statcast()
        .with_batter_babip()
        .build()
    )


def marcel_gb_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Best accuracy pipeline with gradient boosting residual corrections.

    Combines all marcel_full adjusters with ML-predicted residual corrections.
    Uses conservative mode (HR/SB only for batters) to preserve OBP accuracy.

    Performance vs marcel_full (2021-2024 avg):
    - HR correlation: +4% (0.652 → 0.678)
    - SB correlation: +6.5% (0.691 → 0.736)
    - OBP correlation: +3.7% (0.516 → 0.535)
    - Rank accuracy: +4.5% (0.577 → 0.603)
    """
    from fantasy_baseball_manager.pipeline.stages.gb_residual_adjuster import GBResidualConfig

    cfg = config or RegressionConfig()
    gb_config = GBResidualConfig(
        batter_allowed_stats=("hr", "sb"),  # Conservative: only power stats
        pitcher_allowed_stats=("so", "bb"),  # Only K/BB for pitchers
    )
    return (
        PipelineBuilder("marcel_gb", config=cfg)
        .with_park_factors()
        .with_pitcher_normalization()
        .with_pitcher_statcast()
        .with_statcast()
        .with_batter_babip()
        .with_gb_residual(gb_config)
        .build()
    )


def marcel_gb_mle_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Best accuracy pipeline with MLE augmentation for rookies.

    Combines marcel_gb (the best performing pipeline) with ML-based Minor
    League Equivalencies for players with limited MLB history (<200 PA).

    This gives you the best of both worlds:
    - Full marcel_gb accuracy for established players
    - MLE-enhanced projections for rookies and recent call-ups

    Requires a trained MLE model (run scripts/run_mle_evaluation.py).
    """
    from fantasy_baseball_manager.pipeline.stages.gb_residual_adjuster import GBResidualConfig

    cfg = config or RegressionConfig()
    gb_config = GBResidualConfig(
        batter_allowed_stats=("hr", "sb"),
        pitcher_allowed_stats=("so", "bb"),
    )
    return (
        PipelineBuilder("marcel_gb_mle", config=cfg)
        .with_park_factors()
        .with_pitcher_normalization()
        .with_pitcher_statcast()
        .with_statcast()
        .with_batter_babip()
        .with_gb_residual(gb_config)
        .with_mle_for_rookies()
        .build()
    )


def mtl_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """MTL standalone pipeline using neural network for rate prediction.

    Uses trained multi-task learning model to predict stat rates directly
    from Statcast features. Falls back to Marcel rates for players without
    sufficient Statcast data.
    """
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("mtl", config=cfg)
        .with_mtl_rate_computer()
        .build()
    )


def marcel_mtl_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Marcel + MTL blend pipeline.

    Uses Marcel for base rates, then blends with MTL neural network predictions.
    Default blend: 70% Marcel, 30% MTL.
    """
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("marcel_mtl", config=cfg)
        .with_park_factors()
        .with_pitcher_normalization()
        .with_pitcher_statcast()
        .with_statcast()
        .with_batter_babip()
        .with_mtl_blender()
        .build()
    )


def mle_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """ML-based Minor League Equivalencies pipeline.

    Uses trained MLE models to translate minor league stats to MLB equivalents
    for players with limited MLB history (<200 PA). This enables meaningful
    projections for rookies and players recently called up from the minors.

    Players with sufficient MLB history fall back to Marcel rates.
    """
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("mle", config=cfg)
        .with_mle_rate_computer()
        .build()
    )


def steamer_pipeline() -> ExternalProjectionAdapter:
    """Steamer projections from FanGraphs.

    Fetches pre-computed Steamer projections from the FanGraphs API.
    These are industry-standard projections widely used for fantasy baseball.
    """
    cache = create_cache_store()
    source = CachedProjectionSource(
        delegate=FanGraphsProjectionSource(ProjectionSystem.STEAMER),
        cache=cache,
        cache_key="steamer",
        ttl_seconds=604800,  # 7 days
    )
    return ExternalProjectionAdapter(source)


def zips_pipeline() -> ExternalProjectionAdapter:
    """ZiPS projections from FanGraphs.

    Fetches pre-computed ZiPS projections from the FanGraphs API.
    ZiPS is Dan Szymborski's projection system, known for aggressive
    projections on young players.
    """
    cache = create_cache_store()
    source = CachedProjectionSource(
        delegate=FanGraphsProjectionSource(ProjectionSystem.ZIPS),
        cache=cache,
        cache_key="zips",
        ttl_seconds=604800,  # 7 days
    )
    return ExternalProjectionAdapter(source)


PIPELINES: dict[str, Callable[[], Any]] = {
    "marcel_classic": marcel_classic_pipeline,
    "marcel": marcel_pipeline,
    "marcel_full": marcel_full_pipeline,
    "marcel_gb": marcel_gb_pipeline,
    "marcel_gb_mle": marcel_gb_mle_pipeline,
    "mtl": mtl_pipeline,
    "marcel_mtl": marcel_mtl_pipeline,
    "mle": mle_pipeline,
    "steamer": steamer_pipeline,
    "zips": zips_pipeline,
}

_CONFIGURABLE_FACTORIES: dict[str, Callable[[RegressionConfig | None], ProjectionPipeline]] = {
    "marcel": marcel_pipeline,
    "marcel_full": marcel_full_pipeline,
    "marcel_gb": marcel_gb_pipeline,
    "marcel_gb_mle": marcel_gb_mle_pipeline,
    "mtl": mtl_pipeline,
    "marcel_mtl": marcel_mtl_pipeline,
    "mle": mle_pipeline,
}


def build_pipeline(
    name: str,
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Build a pipeline by name, optionally passing a RegressionConfig."""
    if name in _CONFIGURABLE_FACTORIES:
        return _CONFIGURABLE_FACTORIES[name](config)
    if name in PIPELINES:
        return PIPELINES[name]()
    raise ValueError(f"Unknown pipeline: {name}")
