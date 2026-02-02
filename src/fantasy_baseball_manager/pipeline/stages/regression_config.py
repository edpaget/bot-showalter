"""Unified configuration for all regression tuneable parameters."""

from dataclasses import dataclass, field

from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationConfig,
)
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)


@dataclass(frozen=True)
class RegressionConfig:
    """Bundles all tuneable regression parameters into one object.

    Default values replicate current behaviour exactly.
    """

    batting_regression_pa: dict[str, float] = field(default_factory=lambda: dict(BATTING_REGRESSION_PA))
    pitching_regression_outs: dict[str, float] = field(default_factory=lambda: dict(PITCHING_REGRESSION_OUTS))
    pitcher_normalization: PitcherNormalizationConfig = field(default_factory=PitcherNormalizationConfig)
