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


@dataclass(frozen=True)
class TradePlayerDetail:
    player_id: int
    player_name: str
    position: str
    cost: float
    projected_value: float
    surplus: float
    years_remaining: int


@dataclass(frozen=True)
class TradeEvaluation:
    team_a_gives: list[TradePlayerDetail]
    team_b_gives: list[TradePlayerDetail]
    team_a_surplus_delta: float
    team_b_surplus_delta: float
    winner: str  # "team_a" | "team_b" | "even"


@dataclass(frozen=True)
class AdjustedValuation:
    player_id: int
    player_name: str
    player_type: str
    position: str
    original_value: float
    adjusted_value: float
    value_change: float
