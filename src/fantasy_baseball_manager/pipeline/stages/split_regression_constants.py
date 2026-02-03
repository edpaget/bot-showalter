"""Per-stat regression constants for platoon split projections.

Split samples are roughly half the size of full-season samples, so
regression amounts are doubled to account for the additional noise.
"""

from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
)

BATTING_SPLIT_REGRESSION_PA: dict[str, float] = {
    stat: value * 2 for stat, value in BATTING_REGRESSION_PA.items()
}
