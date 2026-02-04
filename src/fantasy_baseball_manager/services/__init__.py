"""Service container for dependency injection."""

from fantasy_baseball_manager.services.cli import cli_context
from fantasy_baseball_manager.services.container import (
    ServiceConfig,
    ServiceContainer,
    get_container,
    set_container,
)

__all__ = [
    "ServiceConfig",
    "ServiceContainer",
    "cli_context",
    "get_container",
    "set_container",
]
