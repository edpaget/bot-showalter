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
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
)
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
    """Modern baseline: per-stat regression + component aging."""
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

    Uses PipelineBuilder to compose: park factors, pitcher normalization,
    pitcher statcast, batter statcast, batter BABIP, rebaseline, and
    component aging.
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
    """Marcel with gradient boosting residual corrections.

    Uses trained ML models to predict and correct Marcel's systematic
    projection errors based on Statcast features.
    """
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("marcel_gb", config=cfg)
        .with_statcast()
        .with_gb_residual()
        .build()
    )


def marcel_full_gb_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Kitchen-sink pipeline with GB residual corrections.

    Combines all marcel_full adjusters with gradient boosting residual
    predictions for maximum accuracy.
    """
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("marcel_full_gb", config=cfg)
        .with_park_factors()
        .with_pitcher_normalization()
        .with_pitcher_statcast()
        .with_statcast()
        .with_batter_babip()
        .with_gb_residual()
        .build()
    )


def marcel_skill_change_pipeline(
    config: RegressionConfig | None = None,
) -> ProjectionPipeline:
    """Marcel with skill change adjustments.

    Detects year-over-year changes in player skills (barrel rate, exit velocity,
    chase rate, whiff rate, sprint speed for batters; fastball velocity, whiff
    rate, ground ball rate for pitchers) and applies targeted projection
    adjustments.
    """
    cfg = config or RegressionConfig()
    return (
        PipelineBuilder("marcel_skill_change", config=cfg)
        .with_statcast()
        .with_skill_change_adjuster()
        .build()
    )


PIPELINES: dict[str, Callable[[], ProjectionPipeline]] = {
    "marcel_classic": marcel_classic_pipeline,
    "marcel": marcel_pipeline,
    "marcel_full": marcel_full_pipeline,
    "marcel_gb": marcel_gb_pipeline,
    "marcel_full_gb": marcel_full_gb_pipeline,
    "marcel_skill_change": marcel_skill_change_pipeline,
}

_CONFIGURABLE_FACTORIES: dict[str, Callable[[RegressionConfig | None], ProjectionPipeline]] = {
    "marcel": marcel_pipeline,
    "marcel_full": marcel_full_pipeline,
    "marcel_gb": marcel_gb_pipeline,
    "marcel_full_gb": marcel_full_gb_pipeline,
    "marcel_skill_change": marcel_skill_change_pipeline,
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
