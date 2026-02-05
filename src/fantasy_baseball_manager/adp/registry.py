"""ADP provider registry for managing multiple ADP sources."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from fantasy_baseball_manager.adp.protocol import ADPSource

if TYPE_CHECKING:
    from fantasy_baseball_manager.adp.models import ADPEntry
    from fantasy_baseball_manager.data.protocol import DataSource

ADPSourceFactory = Callable[[], ADPSource]
ADPDataSourceFactory = Callable[[], "DataSource[ADPEntry]"]

_registry: dict[str, ADPSourceFactory] = {}
_datasource_registry: dict[str, ADPDataSourceFactory] = {}
_defaults_registered = False


def register_source(name: str, factory: ADPSourceFactory) -> None:
    """Register an ADP source factory.

    Args:
        name: The source identifier (e.g., "yahoo", "espn").
        factory: A callable that returns an ADPSource instance.
    """
    _registry[name] = factory


def get_source(name: str) -> ADPSource:
    """Get an ADP source by name.

    Args:
        name: The source identifier.

    Returns:
        An ADPSource instance.

    Raises:
        KeyError: If the source is not registered.
    """
    _ensure_defaults_registered()
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys()))
        raise KeyError(f"Unknown ADP source: {name!r}. Available: {available}")
    return _registry[name]()


def list_sources() -> tuple[str, ...]:
    """List all registered source names.

    Returns:
        Tuple of registered source names.
    """
    _ensure_defaults_registered()
    return tuple(sorted(_registry.keys()))


def reset_registry() -> None:
    """Reset the registry. Used for testing."""
    global _defaults_registered
    _registry.clear()
    _datasource_registry.clear()
    _defaults_registered = False


def _ensure_defaults_registered() -> None:
    """Ensure default providers are registered."""
    global _defaults_registered
    if _defaults_registered:
        return

    from fantasy_baseball_manager.adp.espn_scraper import (
        ESPNADPScraper,
        create_espn_adp_source,
    )
    from fantasy_baseball_manager.adp.scraper import (
        YahooADPScraper,
        create_yahoo_adp_source,
    )

    # Legacy protocol-based sources
    if "yahoo" not in _registry:
        register_source("yahoo", YahooADPScraper)
    if "espn" not in _registry:
        register_source("espn", ESPNADPScraper)

    # New DataSource-based sources
    if "yahoo" not in _datasource_registry:
        register_datasource("yahoo", create_yahoo_adp_source)
    if "espn" not in _datasource_registry:
        register_datasource("espn", create_espn_adp_source)

    _defaults_registered = True


# ---------------------------------------------------------------------------
# New-style DataSource registry
# ---------------------------------------------------------------------------


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
