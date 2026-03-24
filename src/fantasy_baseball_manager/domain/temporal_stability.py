from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class TargetStability:
    target: str
    per_season_r: tuple[tuple[int, float], ...]
    mean_r: float
    std_r: float
    cv: float
    classification: str


@dataclass(frozen=True)
class StabilityResult:
    column_spec: str
    player_type: PlayerType
    seasons: tuple[int, ...]
    target_stabilities: tuple[TargetStability, ...]
