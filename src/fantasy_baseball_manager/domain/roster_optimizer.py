from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetAllocation:
    position: str
    budget: float
    target_tier: int | None
    target_player_names: tuple[str, ...]
