from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerTier:
    player_id: int
    player_name: str
    position: str
    tier: int
    value: float
    rank: int  # within-position rank (1 = best)
