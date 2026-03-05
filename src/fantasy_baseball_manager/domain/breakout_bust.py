from dataclasses import dataclass, field
from enum import StrEnum


class OutcomeLabel(StrEnum):
    BREAKOUT = "breakout"
    BUST = "bust"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class LabelConfig:
    breakout_threshold: int = 30
    bust_threshold: int = -30
    min_adp_rank: int = 300


@dataclass(frozen=True)
class LabeledSeason:
    player_id: int
    season: int
    player_type: str
    adp_rank: int
    adp_pick: float
    actual_value_rank: int
    rank_delta: int
    label: OutcomeLabel


@dataclass(frozen=True)
class BreakoutPrediction:
    player_id: int
    player_name: str
    player_type: str
    position: str
    p_breakout: float
    p_bust: float
    p_neutral: float
    top_features: list[tuple[str, float]] = field(default_factory=list)
