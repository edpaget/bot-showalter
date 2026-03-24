from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class RecommendationWeights:
    value: float = 1.0
    need: float = 0.3
    scarcity: float = 0.4
    tier: float = 0.2
    adp: float = 0.15
    category_balance: float = 0.25
    mock_position: float = 0.3
    mock_availability: float = 0.2


@dataclass(frozen=True)
class Recommendation:
    player_id: int
    player_name: str
    position: str
    value: float
    score: float
    reason: str
    player_type: PlayerType | None = None
