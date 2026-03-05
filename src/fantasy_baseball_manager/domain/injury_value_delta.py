from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InjuryValueDelta:
    player_name: str
    original_value: float
    adjusted_value: float
    value_delta: float
    original_rank: int
    adjusted_rank: int
    rank_change: int
    expected_days_lost: float
