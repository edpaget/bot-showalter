from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedGamesLost:
    player_id: int
    expected_days_lost: float
    p_full_season: float
    confidence: str
