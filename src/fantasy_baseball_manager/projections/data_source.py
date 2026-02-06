"""DataSource implementations for projection data.

Provides new-style DataSource[T] wrappers for projection sources,
following the unified data source pattern.

Usage:
    from fantasy_baseball_manager.projections.data_source import (
        create_batting_projection_source,
        create_pitching_projection_source,
    )

    batting_source = create_batting_projection_source()
    result = batting_source(ALL_PLAYERS)
    if result.is_ok():
        projections = result.unwrap()  # list[BattingProjection]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSource, DataSourceError
from fantasy_baseball_manager.projections.fangraphs import FanGraphsProjectionSource
from fantasy_baseball_manager.projections.models import (
    BattingProjection,
    PitchingProjection,
    ProjectionSystem,
)
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.player.identity import Player
    from fantasy_baseball_manager.projections.protocol import ProjectionSource


class BattingProjectionDataSource:
    """DataSource for batting projections.

    Wraps any ProjectionSource and extracts batting projections,
    implementing the DataSource[BattingProjection] protocol.

    Usage:
        source = BattingProjectionDataSource(FanGraphsProjectionSource())
        result = source(ALL_PLAYERS)
        if result.is_ok():
            projections = result.unwrap()  # list[BattingProjection]
    """

    def __init__(self, source: ProjectionSource) -> None:
        """Initialize the data source.

        Args:
            source: Any ProjectionSource implementation.
        """
        self._source = source

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[BattingProjection]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[BattingProjection]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[BattingProjection] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[BattingProjection]] | Ok[BattingProjection] | Err[DataSourceError]:
        """Fetch batting projections.

        Args:
            query: Must be ALL_PLAYERS. Single-player queries not supported.

        Returns:
            Ok with list of BattingProjection, or Err with DataSourceError.
        """
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported for projection data"))

        try:
            data = self._source.fetch_projections()
            return Ok(list(data.batting))
        except Exception as e:
            return Err(DataSourceError("Failed to fetch batting projections", e))


class PitchingProjectionDataSource:
    """DataSource for pitching projections.

    Wraps any ProjectionSource and extracts pitching projections,
    implementing the DataSource[PitchingProjection] protocol.

    Usage:
        source = PitchingProjectionDataSource(FanGraphsProjectionSource())
        result = source(ALL_PLAYERS)
        if result.is_ok():
            projections = result.unwrap()  # list[PitchingProjection]
    """

    def __init__(self, source: ProjectionSource) -> None:
        """Initialize the data source.

        Args:
            source: Any ProjectionSource implementation.
        """
        self._source = source

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[PitchingProjection]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[PitchingProjection]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[PitchingProjection] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[PitchingProjection]] | Ok[PitchingProjection] | Err[DataSourceError]:
        """Fetch pitching projections.

        Args:
            query: Must be ALL_PLAYERS. Single-player queries not supported.

        Returns:
            Ok with list of PitchingProjection, or Err with DataSourceError.
        """
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported for projection data"))

        try:
            data = self._source.fetch_projections()
            return Ok(list(data.pitching))
        except Exception as e:
            return Err(DataSourceError("Failed to fetch pitching projections", e))


def create_batting_projection_source(
    system: ProjectionSystem = ProjectionSystem.STEAMER,
) -> DataSource[BattingProjection]:
    """Create a DataSource for batting projections from FanGraphs.

    Args:
        system: The projection system to use (default: Steamer).

    Returns:
        DataSource[BattingProjection] that fetches from FanGraphs.

    Usage:
        source = create_batting_projection_source()
        result = source(ALL_PLAYERS)
        if result.is_ok():
            projections = result.unwrap()
    """
    return BattingProjectionDataSource(FanGraphsProjectionSource(system))


def create_pitching_projection_source(
    system: ProjectionSystem = ProjectionSystem.STEAMER,
) -> DataSource[PitchingProjection]:
    """Create a DataSource for pitching projections from FanGraphs.

    Args:
        system: The projection system to use (default: Steamer).

    Returns:
        DataSource[PitchingProjection] that fetches from FanGraphs.

    Usage:
        source = create_pitching_projection_source()
        result = source(ALL_PLAYERS)
        if result.is_ok():
            projections = result.unwrap()
    """
    return PitchingProjectionDataSource(FanGraphsProjectionSource(system))
