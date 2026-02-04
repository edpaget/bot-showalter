"""Data types for minor league statistics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MinorLeagueLevel(Enum):
    """Minor league levels with their MLB Stats API sport IDs."""

    AAA = 11
    AA = 12
    HIGH_A = 13
    SINGLE_A = 14
    ROOKIE = 16

    @classmethod
    def from_sport_id(cls, sport_id: int) -> MinorLeagueLevel:
        """Get level from MLB Stats API sport ID."""
        for level in cls:
            if level.value == sport_id:
                return level
        raise ValueError(f"Unknown sport_id: {sport_id}")

    @classmethod
    def from_code(cls, code: str) -> MinorLeagueLevel:
        """Get level from common abbreviations (AAA, AA, A+, A, Rk)."""
        mapping = {
            "AAA": cls.AAA,
            "AA": cls.AA,
            "A+": cls.HIGH_A,
            "HIGH-A": cls.HIGH_A,
            "A": cls.SINGLE_A,
            "SINGLE-A": cls.SINGLE_A,
            "RK": cls.ROOKIE,
            "ROOKIE": cls.ROOKIE,
        }
        normalized = code.upper().strip()
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(f"Unknown level code: {code}")

    @property
    def display_name(self) -> str:
        """Human-readable level name."""
        names = {
            self.AAA: "AAA",
            self.AA: "AA",
            self.HIGH_A: "A+",
            self.SINGLE_A: "A",
            self.ROOKIE: "Rookie",
        }
        return names[self]


@dataclass(frozen=True)
class MinorLeagueBatterSeasonStats:
    """Batting statistics for a single minor league season."""

    player_id: str
    name: str
    season: int
    age: int
    level: MinorLeagueLevel
    team: str
    league: str
    pa: int
    ab: int
    h: int
    singles: int
    doubles: int
    triples: int
    hr: int
    rbi: int
    r: int
    bb: int
    so: int
    hbp: int
    sf: int
    sb: int
    cs: int
    avg: float
    obp: float
    slg: float

    @property
    def sport_id(self) -> int:
        """MLB Stats API sport ID for this level."""
        return self.level.value


@dataclass(frozen=True)
class MinorLeaguePitcherSeasonStats:
    """Pitching statistics for a single minor league season."""

    player_id: str
    name: str
    season: int
    age: int
    level: MinorLeagueLevel
    team: str
    league: str
    g: int
    gs: int
    ip: float
    w: int
    losses: int
    sv: int
    h: int
    r: int
    er: int
    hr: int
    bb: int
    so: int
    hbp: int
    era: float
    whip: float

    @property
    def sport_id(self) -> int:
        """MLB Stats API sport ID for this level."""
        return self.level.value

    @property
    def batters_faced(self) -> float:
        """Approximate batters faced (for rate calculations)."""
        # BF ≈ IP * 3 + H + BB + HBP
        return self.ip * 3 + self.h + self.bb + self.hbp


@dataclass(frozen=True)
class MLEPrediction:
    """Predicted MLB-equivalent rates from minor league stats."""

    player_id: str
    source_season: int
    source_level: MinorLeagueLevel
    source_pa: int
    predicted_rates: dict[str, float]
    confidence: float
