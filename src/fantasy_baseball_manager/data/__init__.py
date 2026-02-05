"""Data source protocols and types."""

from typing import TYPE_CHECKING

from fantasy_baseball_manager.data.protocol import (
    ALL_PLAYERS,
    DataSource,
    DataSourceError,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import (  # noqa: TC004 - only defined in TYPE_CHECKING
        DataSourceResult,
        Query,
    )

__all__ = [
    "ALL_PLAYERS",
    "DataSource",
    "DataSourceError",
    "DataSourceResult",
    "Query",
]
