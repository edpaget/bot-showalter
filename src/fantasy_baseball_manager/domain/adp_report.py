from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class ValueOverADP:
    player_id: int
    player_name: str
    player_type: PlayerType
    position: str
    adp_positions: str
    zar_rank: int
    zar_value: float
    adp_rank: int
    adp_pick: float
    rank_delta: int
    provider: str


@dataclass(frozen=True)
class ValueOverADPReport:
    season: int
    system: str
    version: str
    provider: str
    buy_targets: list[ValueOverADP]
    avoid_list: list[ValueOverADP]
    unranked_valuable: list[ValueOverADP]
    n_matched: int
