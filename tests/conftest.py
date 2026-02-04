"""Shared pytest fixtures for test modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

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
