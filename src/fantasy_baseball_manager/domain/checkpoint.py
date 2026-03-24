from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.experiment import TargetResult
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class FeatureCheckpoint:
    name: str
    model: str
    player_type: PlayerType
    feature_columns: list[str]
    params: dict[str, Any]
    target_results: dict[str, TargetResult]
    experiment_id: int
    created_at: str
    notes: str = ""
