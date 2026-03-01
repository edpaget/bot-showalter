from dataclasses import dataclass


@dataclass(frozen=True)
class KeeperCost:
    player_id: int
    season: int
    league: str
    cost: float
    source: str
    years_remaining: int = 1
    id: int | None = None
    loaded_at: str | None = None
