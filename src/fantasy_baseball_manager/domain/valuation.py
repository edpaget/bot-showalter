from dataclasses import dataclass


@dataclass(frozen=True)
class Valuation:
    player_id: int
    season: int
    system: str
    version: str
    projection_system: str
    projection_version: str
    player_type: str
    position: str
    value: float
    rank: int
    category_scores: dict[str, float]
    id: int | None = None
    loaded_at: str | None = None


@dataclass(frozen=True)
class PlayerValuation:
    player_name: str
    system: str
    version: str
    projection_system: str
    projection_version: str
    player_type: str
    position: str
    value: float
    rank: int
    category_scores: dict[str, float]


@dataclass(frozen=True)
class ValuationAccuracy:
    player_id: int
    player_name: str
    player_type: str
    position: str
    predicted_value: float
    actual_value: float
    surplus: float  # predicted - actual (positive = overpaid)
    predicted_rank: int
    actual_rank: int


@dataclass(frozen=True)
class ValuationEvalResult:
    system: str
    version: str
    season: int
    value_mae: float
    rank_correlation: float  # Spearman rho
    n: int
    players: list[ValuationAccuracy]
