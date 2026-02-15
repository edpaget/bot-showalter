import inspect
from typing import Any

from fantasy_baseball_manager.models.protocols import ProjectionModel
from fantasy_baseball_manager.models.registry import get


def create_model(name: str, **kwargs: Any) -> ProjectionModel:
    """Look up a model class by name and instantiate it, forwarding matching kwargs."""
    cls = get(name)
    sig = inspect.signature(cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered)
