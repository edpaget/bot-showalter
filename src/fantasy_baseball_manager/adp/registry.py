"""ADP provider registry for managing multiple ADP data sources."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.adp.models import ADPEntry
    from fantasy_baseball_manager.data.protocol import DataSource

ADPDataSourceFactory = Callable[[], "DataSource[ADPEntry]"]

_datasource_registry: dict[str, ADPDataSourceFactory] = {}
_defaults_registered = False


def register_datasource(name: str, factory: ADPDataSourceFactory) -> None:
    """Register a DataSource[ADPEntry] factory.

    Args:
        name: The source identifier (e.g., "yahoo", "espn").
        factory: A callable that returns a DataSource[ADPEntry] instance.
    """
    _datasource_registry[name] = factory


def get_datasource(name: str) -> DataSource[ADPEntry]:
    """Get a DataSource[ADPEntry] by name.

    Args:
        name: The source identifier.

    Returns:
        A DataSource[ADPEntry] instance.

    Raises:
        KeyError: If the source is not registered.
    """
    _ensure_defaults_registered()
    if name not in _datasource_registry:
        available = ", ".join(sorted(_datasource_registry.keys()))
        raise KeyError(f"Unknown ADP data source: {name!r}. Available: {available}")
    return _datasource_registry[name]()


def list_datasources() -> tuple[str, ...]:
    """List all registered DataSource names.

    Returns:
        Tuple of registered data source names.
    """
    _ensure_defaults_registered()
    return tuple(sorted(_datasource_registry.keys()))


def reset_registry() -> None:
    """Reset the registry. Used for testing."""
    global _defaults_registered
    _datasource_registry.clear()
    _defaults_registered = False


def _ensure_defaults_registered() -> None:
    """Ensure default providers are registered."""
    global _defaults_registered
    if _defaults_registered:
        return

    from fantasy_baseball_manager.adp.espn_scraper import create_espn_adp_source
    from fantasy_baseball_manager.adp.scraper import create_yahoo_adp_source

    if "yahoo" not in _datasource_registry:
        register_datasource("yahoo", create_yahoo_adp_source)
    if "espn" not in _datasource_registry:
        register_datasource("espn", create_espn_adp_source)

    _defaults_registered = True
