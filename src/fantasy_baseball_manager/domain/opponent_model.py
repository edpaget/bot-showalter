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
class PositionRun:
    """A detected run of picks at the same position within a recent window."""

    position: str
    pick_numbers: tuple[int, ...]
    run_length: int
    remaining_supply: int
    urgency: str  # "critical" | "developing"
