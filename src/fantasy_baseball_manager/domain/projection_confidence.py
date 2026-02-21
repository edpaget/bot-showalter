from dataclasses import dataclass
from enum import StrEnum


class VarianceClassification(StrEnum):
    SAFE_CONSENSUS = "safe_consensus"
    UPSIDE_GAMBLE = "upside_gamble"
    RISKY_AVOID = "risky_avoid"
    HIDDEN_UPSIDE = "hidden_upside"
    KNOWN_QUANTITY = "known_quantity"


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


@dataclass(frozen=True)
class ClassifiedPlayer:
    player: PlayerConfidence
    classification: VarianceClassification
    adp_rank: int | None
    value_rank: int
    risk_reward_score: float
