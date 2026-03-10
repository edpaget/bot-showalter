from dataclasses import dataclass


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
