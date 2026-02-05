"""Protocol for projection sources."""

from typing import Protocol

from fantasy_baseball_manager.projections.models import ProjectionData


class ProjectionSource(Protocol):
    """Protocol for external projection data sources.

    Any class that implements fetch_projections() -> ProjectionData satisfies
    this protocol. This enables dependency injection and easier testing.
    """

    def fetch_projections(self) -> ProjectionData:
        """Fetch projections from the source.

        Returns:
            ProjectionData containing batting and pitching projections.
        """
        ...
