from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetAllocation:
    position: str
    budget: float
    target_tier: int | None
    target_player_names: tuple[str, ...]


@dataclass(frozen=True)
class RoundTarget:
    round: int
    pick_number: int
    recommended_position: str
    target_tier: int | None
    expected_value: float
    alternative_positions: tuple[str, ...]


@dataclass(frozen=True)
class SnakeDraftPlan:
    draft_slot: int
    teams: int
    rounds: list[RoundTarget]
