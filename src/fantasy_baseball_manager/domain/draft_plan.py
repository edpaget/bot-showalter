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
