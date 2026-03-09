from dataclasses import dataclass


@dataclass(frozen=True)
class TeamNeeds:
    team_idx: int
    team_name: str | None
    filled: dict[str, int]
    unfilled: dict[str, int]
    total_value: float


@dataclass(frozen=True)
class LeagueNeeds:
    teams: list[TeamNeeds]
    demand_by_position: dict[str, int]
    supply_by_position: dict[str, int]
    scarcity_ratio: dict[str, float]


@dataclass(frozen=True)
class ThreatAssessment:
    """Assessment of how likely a player is to be taken before the user's next pick."""

    player_id: int
    player_name: str
    position: str
    value: float
    adp: float | None
    picks_until_user_next: int
    teams_needing_position: int
    threat_level: str  # "safe" | "at-risk" | "likely-gone"


@dataclass(frozen=True)
class PositionRun:
    """A detected run of picks at the same position within a recent window."""

    position: str
    pick_numbers: tuple[int, ...]
    run_length: int
    remaining_supply: int
    urgency: str  # "critical" | "developing"
