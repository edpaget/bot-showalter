from dataclasses import dataclass
from enum import Enum


class StatCategory(Enum):
    HR = "HR"
    R = "R"
    RBI = "RBI"
    SB = "SB"
    OBP = "OBP"
    W = "W"
    K = "K"
    ERA = "ERA"
    WHIP = "WHIP"
    NSVH = "NSVH"


@dataclass(frozen=True)
class LeagueSettings:
    team_count: int
    batting_categories: tuple[StatCategory, ...]
    pitching_categories: tuple[StatCategory, ...]


@dataclass(frozen=True)
class CategoryValue:
    category: StatCategory
    raw_stat: float
    value: float


@dataclass(frozen=True)
class PlayerValue:
    player_id: str
    name: str
    category_values: tuple[CategoryValue, ...]
    total_value: float
    position_type: str = ""


@dataclass(frozen=True)
class SGPDenominators:
    denominators: dict[StatCategory, float]
