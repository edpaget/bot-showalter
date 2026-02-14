from collections.abc import Callable

from fantasy_baseball_manager.models.protocols import ProjectionModel

_REGISTRY: dict[str, Callable[[], ProjectionModel]] = {}


def register(name: str) -> Callable[[type], type]:
    """Decorator that registers a model class under a name."""

    def decorator(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get(name: str) -> ProjectionModel:
    """Instantiate and return a registered model."""
    if name not in _REGISTRY:
        raise KeyError(f"'{name}': no model registered with this name")
    return _REGISTRY[name]()


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY)


def _clear() -> None:
    """Clear the registry. For testing only."""
    _REGISTRY.clear()
