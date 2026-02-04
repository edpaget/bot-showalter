"""Shared CLI utilities for ServiceContainer management."""

from collections.abc import Generator
from contextlib import contextmanager

from fantasy_baseball_manager.services.container import ServiceConfig, ServiceContainer, set_container


@contextmanager
def cli_context(
    league_id: str | None = None,
    season: int | None = None,
    no_cache: bool = False,
) -> Generator[None]:
    """Context manager that sets up ServiceContainer with CLI overrides.

    If a container is already set (e.g., by tests), uses the existing container
    and doesn't reset it on exit. This allows tests to inject fake dependencies.
    """
    from fantasy_baseball_manager.services.container import _container

    if _container is not None:
        # Container already set (test mode) — use it without changes
        yield
        return

    config = ServiceConfig(no_cache=no_cache, league_id=league_id, season=season)
    container = ServiceContainer(config)
    set_container(container)
    try:
        yield
    finally:
        set_container(None)
