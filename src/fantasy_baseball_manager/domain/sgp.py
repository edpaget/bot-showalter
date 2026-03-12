from dataclasses import dataclass
from enum import StrEnum


class DenominatorMethod(StrEnum):
    MEAN_GAP = "mean_gap"
    REGRESSION = "regression"


@dataclass(frozen=True)
class SgpSeasonDenominator:
    category: str
    season: int
    denominator: float
    num_teams: int


@dataclass(frozen=True)
class SgpDenominators:
    per_season: tuple[SgpSeasonDenominator, ...]
    averages: dict[str, float]  # category → mean denominator across seasons
