from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftPlanTarget:
    """A position-targeting recommendation for a range of draft rounds."""

    round_range: tuple[int, int]  # (start, end) inclusive
    position: str
    confidence: float  # 0.0-1.0
    example_players: list[str]  # top 3 most-frequently-drafted names


@dataclass(frozen=True)
class DraftPlan:
    """Actionable draft guidance distilled from mock draft simulations."""

    slot: int
    teams: int
    strategy_name: str
    targets: list[DraftPlanTarget]
    n_simulations: int
    avg_roster_value: float


@dataclass(frozen=True)
class AvailabilityWindow:
    """Summary of when a player is typically drafted across simulations."""

    player_id: int
    player_name: str
    position: str
    earliest_pick: float  # 5th percentile
    median_pick: float  # 50th percentile
    latest_pick: float  # 95th percentile
    available_at_user_pick: float  # P(available) at a specific user pick


@dataclass(frozen=True)
class PickAvailability:
    """Availability probability at a specific pick."""

    round: int
    pick: int
    probability: float


@dataclass(frozen=True)
class PlayerAvailabilityCurve:
    """Round-by-round availability curve for a single player."""

    player_id: int
    player_name: str
    position: str
    pick_availabilities: list[PickAvailability]
