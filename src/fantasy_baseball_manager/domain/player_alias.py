from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class PlayerAlias:
    alias_name: str
    player_id: int
    player_type: PlayerType | None = None
    source: str | None = None
    active_from: int | None = None
    active_to: int | None = None
    id: int | None = None
