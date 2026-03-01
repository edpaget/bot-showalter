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


@dataclass(frozen=True)
class KeeperDecision:
    player_id: int
    player_name: str
    position: str
    cost: float
    projected_value: float
    surplus: float
    years_remaining: int
    recommendation: str  # "keep" | "release"
