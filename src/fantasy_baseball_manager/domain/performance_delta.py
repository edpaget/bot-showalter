from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerStatDelta:
    player_id: int
    player_name: str
    stat_name: str
    actual: float
    expected: float
    delta: float
    performance_delta: float
    percentile: float
