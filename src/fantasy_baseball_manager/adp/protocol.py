"""ADP source protocol for dependency injection."""

from typing import Protocol

from fantasy_baseball_manager.adp.models import ADPData


class ADPSource(Protocol):
    """Protocol for ADP data sources.

    Any class that implements fetch_adp() -> ADPData satisfies this protocol.
    This enables dependency injection and easier testing.
    """

    def fetch_adp(self) -> ADPData:
        """Fetch ADP data from the source.

        Returns:
            ADPData containing player ADP entries.
        """
        ...
