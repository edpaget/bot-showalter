"""Unified data source protocol.

Provides a single callable signature for all data sources. Year comes from context.

Usage:
    # Batch query for all players in context's year
    result = batting_source(ALL_PLAYERS)
    if result.is_ok():
        stats = result.unwrap()  # Type checker knows this is Sequence[T]

    # Single player query (if supported)
    result = batting_source(player)
    if result.is_ok():
        stat = result.unwrap()  # Type checker knows this is T

Return type behavior (with full type inference):
    - source(ALL_PLAYERS) -> Ok[Sequence[T]] | Err[DataSourceError]
    - source(player) -> Ok[T] | Err[DataSourceError]
    - source([p1, p2]) -> Ok[Sequence[T]] | Err[DataSourceError]

Implementation flexibility:
    DataSources are NOT required to support all query types. They can return
    Err(DataSourceError("Single player queries not supported")) for unsupported
    variants. This allows batch-only sources to exist without forcing inefficient
    single-player lookups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, final, overload

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.player.identity import Player
    from fantasy_baseball_manager.result import Err, Ok

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


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


class DataSource(Protocol[T_co]):
    """Protocol for data sources with overloaded return types.

    Using @overload allows the type checker to infer the correct return type
    based on the query type, eliminating the need for casts.

    Uses list[T] instead of Sequence[T] to avoid generic invariance issues
    (Ok[list[T]] doesn't match Ok[Sequence[T]] in the type system).
    """

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[T_co]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[T_co]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[T_co] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[T_co]] | Ok[T_co] | Err[DataSourceError]: ...


# Simpler type for batch-only sources (most common case)
# This avoids Protocol complexity when you only need ALL_PLAYERS queries
class BatchDataSource(Protocol[T_co]):
    """Protocol for batch-only data sources.

    Simpler than DataSource - only supports ALL_PLAYERS queries.
    Returns list[T] directly, so type checker knows the result type.
    """

    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[T_co]] | Err[DataSourceError]: ...


# Legacy type aliases for backward compatibility
if TYPE_CHECKING:
    # Query type for DataSource
    Query = type[ALL_PLAYERS] | Player | list[Player]

    # Result type returned by DataSource (union form, less precise)
    DataSourceResult = Ok[Sequence[T]] | Ok[T] | Err[DataSourceError]
