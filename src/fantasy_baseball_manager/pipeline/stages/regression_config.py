"""Unified configuration for all regression tuneable parameters."""

from dataclasses import dataclass, field

from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationConfig,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_statcast_adjuster import (
    PitcherStatcastConfig,
)
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)
from fantasy_baseball_manager.pipeline.stages.split_regression_constants import (
    BATTING_SPLIT_REGRESSION_PA,
)


@dataclass(frozen=True)
class PlatoonConfig:
    """Configuration for platoon split projections."""

    pct_vs_rhp: float = 0.72
    pct_vs_lhp: float = 0.28
    batting_split_regression_pa: dict[str, float] = field(
        default_factory=lambda: dict(BATTING_SPLIT_REGRESSION_PA)
    )


@dataclass(frozen=True)
class RegressionConfig:
    """Bundles all tuneable regression parameters into one object.

    Default values replicate current behaviour exactly.
    """

    batting_regression_pa: dict[str, float] = field(default_factory=lambda: dict(BATTING_REGRESSION_PA))
    pitching_regression_outs: dict[str, float] = field(default_factory=lambda: dict(PITCHING_REGRESSION_OUTS))
    pitcher_normalization: PitcherNormalizationConfig = field(default_factory=PitcherNormalizationConfig)
    pitcher_statcast: PitcherStatcastConfig = field(default_factory=PitcherStatcastConfig)
    platoon: PlatoonConfig = field(default_factory=PlatoonConfig)
