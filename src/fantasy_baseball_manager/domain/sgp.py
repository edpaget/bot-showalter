from dataclasses import dataclass, field
from enum import StrEnum

# Maps category key → (avg_team_rate, avg_team_volume)
RepresentativeTeamTotals = dict[str, tuple[float, float]]


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
    representative_team: RepresentativeTeamTotals = field(default_factory=dict)
