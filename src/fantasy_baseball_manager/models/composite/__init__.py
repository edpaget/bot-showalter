from fantasy_baseball_manager.models.composite.model import CompositeModel  # noqa: F401
from fantasy_baseball_manager.models.registry import register_alias

for _alias in ("composite-mle", "composite-statcast", "composite-full"):
    register_alias(_alias, "composite")
