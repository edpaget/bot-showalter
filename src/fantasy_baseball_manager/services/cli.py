"""Shared CLI utilities for ServiceContainer management."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from fantasy_baseball_manager.context import init_context
from fantasy_baseball_manager.services.container import ServiceConfig, ServiceContainer, set_container


@contextmanager
def cli_context(
    league_id: str | None = None,
    season: int | None = None,
    no_cache: bool = False,
    refresh: bool = False,
) -> Generator[None]:
    """Context manager that sets up ServiceContainer with CLI overrides.

    Initializes both the ambient Context (for data sources) and the ServiceContainer
    (for legacy dependency injection).

    If a container is already set (e.g., by tests), uses the existing container
    and doesn't reset it on exit. This allows tests to inject fake dependencies.

    Args:
        league_id: Override the league ID from config file.
        season: Override the season from config file.
        no_cache: Disable caching for all services.
        refresh: Skip cache reads but allow writes (refresh mode).
    """
    from fantasy_baseball_manager.services.container import _container

    if _container is not None:
        # Container already set (test mode) — use it without changes
        yield
        return

    config = ServiceConfig(no_cache=no_cache, league_id=league_id, season=season)
    container = ServiceContainer(config)
    set_container(container)

    # Initialize ambient context for data sources
    # Get season from container's resolved config
    resolved_season = int(str(container.app_config["league.season"]))
    db_path = Path(str(container.app_config["cache.db_path"])).expanduser()

    init_context(
        year=resolved_season,
        cache_enabled=not no_cache,
        cache_invalidated=refresh,
        db_path=db_path,
    )

    try:
        yield
    finally:
        set_container(None)
