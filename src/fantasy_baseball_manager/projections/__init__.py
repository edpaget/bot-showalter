"""External projection sources (Steamer, ZiPS, etc.)."""

from fantasy_baseball_manager.projections.data_source import (
    BattingProjectionDataSource,
    PitchingProjectionDataSource,
    create_batting_projection_source,
    create_pitching_projection_source,
)
from fantasy_baseball_manager.projections.fangraphs import FanGraphsProjectionSource
from fantasy_baseball_manager.projections.models import (
    BattingProjection,
    PitchingProjection,
    ProjectionData,
    ProjectionSystem,
)
from fantasy_baseball_manager.projections.protocol import ProjectionSource

__all__ = [
    "BattingProjection",
    "BattingProjectionDataSource",
    "FanGraphsProjectionSource",
    "PitchingProjection",
    "PitchingProjectionDataSource",
    "ProjectionData",
    "ProjectionSource",
    "ProjectionSystem",
    "create_batting_projection_source",
    "create_pitching_projection_source",
]
