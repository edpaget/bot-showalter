"""MTL-specific model store with batter/pitcher class dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.registry.base_store import BaseModelStore

if TYPE_CHECKING:
    from fantasy_baseball_manager.ml.mtl.model import (
        MultiTaskBatterModel,
        MultiTaskPitcherModel,
    )


@dataclass
class MTLBaseModelStore(BaseModelStore):
    """Model store for MTL models that dispatches to the correct class on load.

    MTL has separate batter and pitcher model classes (MultiTaskBatterModel,
    MultiTaskPitcherModel) that each have their own from_params() classmethod.
    This subclass overrides load to dispatch based on player_type.
    """

    def load_batter(self, name: str) -> MultiTaskBatterModel:
        """Load a batter model, returning the correctly typed class."""
        from fantasy_baseball_manager.ml.mtl.model import MultiTaskBatterModel

        params = self.load_params(name, "batter")
        return MultiTaskBatterModel.from_params(params)

    def load_pitcher(self, name: str) -> MultiTaskPitcherModel:
        """Load a pitcher model, returning the correctly typed class."""
        from fantasy_baseball_manager.ml.mtl.model import MultiTaskPitcherModel

        params = self.load_params(name, "pitcher")
        return MultiTaskPitcherModel.from_params(params)

    def save_model(
        self,
        model: MultiTaskBatterModel | MultiTaskPitcherModel,
        name: str,
        player_type: str,
        *,
        version: int = 1,
    ) -> Any:
        """Save an MTL model, extracting metadata from the model."""
        return self.save_params(
            model.get_params(),
            name,
            player_type,
            training_years=model.training_years,
            stats=list(model.STATS),
            feature_names=model.feature_names,
            metrics={"validation_metrics": model.validation_metrics} if model.validation_metrics else {},
            version=version,
        )
