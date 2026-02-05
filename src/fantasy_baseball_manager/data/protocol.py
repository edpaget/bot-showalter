"""Unified data source protocol.

Provides a single callable signature for all data sources. Year comes from context.

Usage:
    # Batch query for all players in context's year
    result = batting_source(ALL_PLAYERS)
    if result.is_ok():
        stats = result.unwrap()

    # Single player query (if supported)
    result = batting_source(player)
    if result.is_err():
        # Source may not support single-player queries
        error = result.unwrap_err()

Return type behavior:
    - source(ALL_PLAYERS) -> Result[Sequence[T], DataSourceError]
    - source(player) -> Result[T, DataSourceError]
    - source([p1, p2]) -> Result[Sequence[T], DataSourceError]

Implementation flexibility:
    DataSources are NOT required to support all query types. They can return
    Err(DataSourceError("Single player queries not supported")) for unsupported
    variants. This allows batch-only sources to exist without forcing inefficient
    single-player lookups.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar, final

if TYPE_CHECKING:
    from fantasy_baseball_manager.player.identity import Player
    from fantasy_baseball_manager.result import Err, Ok

T = TypeVar("T")


@final
class ALL_PLAYERS:
    """Sentinel: fetch all players for context's year.

    Usage:
        result = source(ALL_PLAYERS)

    Note: This is a class used as a sentinel, not instantiated.
    """

    def __new__(cls) -> ALL_PLAYERS:
        raise TypeError("ALL_PLAYERS is a sentinel and should not be instantiated")


class DataSourceError(Exception):
    """Base error for data source failures.

    Attributes:
        message: Human-readable error description.
        cause: Optional underlying exception that caused this error.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message}: {self.cause}"
        return self.message


# Type aliases need to be in TYPE_CHECKING block or use forward references
# to avoid runtime errors when Player hasn't been imported yet
if TYPE_CHECKING:
    # Query type for DataSource
    Query = type[ALL_PLAYERS] | Player | Sequence[Player]

    # Result type returned by DataSource
    # For ALL_PLAYERS and Sequence[Player] queries: Sequence[T]
    # For single Player query: T
    DataSourceResult = Ok[Sequence[T]] | Ok[T] | Err[DataSourceError]

    # The unified DataSource type
    # A callable that accepts a query and returns a Result
    DataSource = Callable[[Query], DataSourceResult[T]]
