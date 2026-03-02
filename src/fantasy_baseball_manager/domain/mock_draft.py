from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BotStrategy(StrEnum):
    ADP_BASED = "adp_based"
    BEST_VALUE = "best_value"
    POSITIONAL_NEED = "positional_need"
    RANDOM = "random"


@dataclass(frozen=True)
class DraftPick:
    round: int
    pick: int
    team_idx: int
    player_id: int
    player_name: str
    position: str
    value: float


@dataclass(frozen=True)
class DraftResult:
    picks: list[DraftPick]
    rosters: dict[int, list[DraftPick]]
    snake: bool
