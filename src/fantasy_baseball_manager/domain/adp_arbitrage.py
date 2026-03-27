from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class FallingPlayer:
    player_id: int
    player_name: str
    position: str
    adp: float
    current_pick: int
    picks_past_adp: float
    value: float
    value_rank: int
    arbitrage_score: float
    player_type: PlayerType | None = None


@dataclass(frozen=True)
class ReachPick:
    player_id: int
    player_name: str
    position: str
    adp: float
    pick_number: int
    picks_ahead_of_adp: float
    drafter_team: int
    player_type: PlayerType | None = None


@dataclass(frozen=True)
class ArbitrageReport:
    current_pick: int
    falling: list[FallingPlayer]
    reaches: list[ReachPick]
