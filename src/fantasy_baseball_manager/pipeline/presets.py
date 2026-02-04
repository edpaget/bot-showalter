"""Pre-configured projection pipelines.

Available pipelines:
- marcel_classic: Original Marcel with uniform aging
- marcel: Modern baseline with per-stat regression and component aging
- marcel_full: Kitchen-sink with park factors, Statcast, and BABIP adjustments
- marcel_gb: marcel_full + gradient boosting residual corrections (best accuracy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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


PIPELINES: dict[str, Callable[[], ProjectionPipeline]] = {
    "marcel_classic": marcel_classic_pipeline,
    "marcel": marcel_pipeline,
    "marcel_full": marcel_full_pipeline,
    "marcel_gb": marcel_gb_pipeline,
}

_CONFIGURABLE_FACTORIES: dict[str, Callable[[RegressionConfig | None], ProjectionPipeline]] = {
    "marcel": marcel_pipeline,
    "marcel_full": marcel_full_pipeline,
    "marcel_gb": marcel_gb_pipeline,
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
