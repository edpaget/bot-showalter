from dataclasses import dataclass


@dataclass(frozen=True)
class StatSpread:
    stat: str
    min_value: float
    max_value: float
    mean: float
    std: float
    cv: float
    systems: dict[str, float]


@dataclass(frozen=True)
class PlayerConfidence:
    player_id: int
    player_name: str
    player_type: str
    position: str
    spreads: list[StatSpread]
    overall_cv: float
    agreement_level: str


@dataclass(frozen=True)
class ConfidenceReport:
    season: int
    systems: list[str]
    players: list[PlayerConfidence]
