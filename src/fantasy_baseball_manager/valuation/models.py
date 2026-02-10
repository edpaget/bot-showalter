from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection


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


class ScoringStyle(Enum):
    H2H_EACH_CATEGORY = "h2h_each_category"
    H2H_MOST_CATEGORIES = "h2h_most_categories"
    ROTO = "roto"
    H2H_POINTS = "h2h_points"


@dataclass(frozen=True)
class LeagueSettings:
    team_count: int
    batting_categories: tuple[StatCategory, ...]
    pitching_categories: tuple[StatCategory, ...]
    scoring_style: ScoringStyle = ScoringStyle.H2H_EACH_CATEGORY


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


@dataclass(frozen=True)
class ValuationResult:
    values: list[PlayerValue]
    categories: tuple[StatCategory, ...]
    label: str


class Valuator(Protocol):
    def valuate_batting(
        self,
        projections: list[BattingProjection],
        categories: tuple[StatCategory, ...],
    ) -> ValuationResult: ...

    def valuate_pitching(
        self,
        projections: list[PitchingProjection],
        categories: tuple[StatCategory, ...],
    ) -> ValuationResult: ...
