"""Shared pytest fixtures for test modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from fantasy_baseball_manager.context import Context, init_context
from fantasy_baseball_manager.services import ServiceContainer, set_container


@pytest.fixture
def reset_service_container() -> Generator[None]:
    """Reset the global ServiceContainer after the test.

    This fixture is NOT autouse - use it explicitly in tests that need cleanup.
    Individual test modules that use ServiceContainer already have their own
    autouse `reset_container` fixture for backwards compatibility.
    """
    yield
    set_container(None)


@pytest.fixture
def mock_container(reset_service_container: None) -> Generator[ServiceContainer]:
    """Create and install a mock ServiceContainer.

    The container is created with no dependencies, allowing tests to inject
    specific fakes as needed. The container is automatically cleaned up after
    the test.

    Usage:
        def test_something(mock_container: ServiceContainer) -> None:
            # Container is installed but empty - inject what you need
            set_container(ServiceContainer(data_source=my_fake_source))
            # ... test code ...
    """
    container = ServiceContainer()
    set_container(container)
    yield container
    # Cleanup handled by reset_service_container dependency


@pytest.fixture
def test_context(tmp_path: Path) -> Generator[Context]:
    """Initialize a test context with a temporary database path.

    This fixture initializes the ambient Context system with a temporary
    database for cache operations, ensuring test isolation.

    The context is set to year 2025 by default with caching enabled.

    Usage:
        def test_data_source(test_context: Context) -> None:
            # Context is initialized with tmp db path
            assert get_context().year == 2025
    """
    ctx = init_context(
        year=2025,
        cache_enabled=True,
        cache_invalidated=False,
        db_path=tmp_path / "test_cache.db",
    )
    yield ctx


@pytest.fixture
def no_cache_context(tmp_path: Path) -> Generator[Context]:
    """Initialize a test context with caching disabled.

    Useful for testing data source behavior without cache interference.

    Usage:
        def test_uncached_fetch(no_cache_context: Context) -> None:
            # Cache is disabled
            assert not get_context().cache_enabled
    """
    ctx = init_context(
        year=2025,
        cache_enabled=False,
        cache_invalidated=False,
        db_path=tmp_path / "test_cache.db",
    )
    yield ctx
