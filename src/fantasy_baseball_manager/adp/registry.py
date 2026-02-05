"""ADP provider registry for managing multiple ADP sources."""

from collections.abc import Callable

from fantasy_baseball_manager.adp.protocol import ADPSource

ADPSourceFactory = Callable[[], ADPSource]

_registry: dict[str, ADPSourceFactory] = {}
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
    _defaults_registered = False


def _ensure_defaults_registered() -> None:
    """Ensure default providers are registered."""
    global _defaults_registered
    if _defaults_registered:
        return

    from fantasy_baseball_manager.adp.espn_scraper import ESPNADPScraper
    from fantasy_baseball_manager.adp.scraper import YahooADPScraper

    if "yahoo" not in _registry:
        register_source("yahoo", YahooADPScraper)
    if "espn" not in _registry:
        register_source("espn", ESPNADPScraper)

    _defaults_registered = True
