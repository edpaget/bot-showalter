from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class ReplacementProfile:
    position: str
    player_type: PlayerType
    stat_line: dict[str, float]
