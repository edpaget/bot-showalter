"""Data source protocols and types."""

from typing import TYPE_CHECKING

from fantasy_baseball_manager.data.protocol import (
    ALL_PLAYERS,
    DataSourceError,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import (
        DataSource,
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
