from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class YahooPlayerMap:
    yahoo_player_key: str
    player_id: int
    player_type: PlayerType
    yahoo_name: str
    yahoo_team: str
    yahoo_positions: str
    id: int | None = None
