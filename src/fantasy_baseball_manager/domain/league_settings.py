from dataclasses import dataclass, field
from enum import StrEnum


class StatType(StrEnum):
    COUNTING = "counting"
    RATE = "rate"


class Direction(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


class LeagueFormat(StrEnum):
    H2H_CATEGORIES = "h2h_categories"
    ROTO = "roto"


@dataclass(frozen=True)
class CategoryConfig:
    key: str
    name: str
    stat_type: StatType
    direction: Direction
    numerator: str | None = None
    denominator: str | None = None


@dataclass(frozen=True)
class EligibilityRules:
    batter_min_games: int = 5
    sp_min_starts: int = 3
    rp_min_relief: int = 5
    carryover_seasons: int = 1


@dataclass(frozen=True)
class LeagueSettings:
    name: str
    format: LeagueFormat
    teams: int
    budget: int
    roster_batters: int
    roster_pitchers: int
    batting_categories: tuple[CategoryConfig, ...]
    pitching_categories: tuple[CategoryConfig, ...]
    roster_util: int = 0
    positions: dict[str, int] = field(default_factory=dict)
    pitcher_positions: dict[str, int] = field(default_factory=dict)
    eligibility: EligibilityRules = field(default_factory=EligibilityRules)
