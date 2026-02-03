from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.draft.simulation import generate_snake_order
from fantasy_baseball_manager.keeper.models import SlotCost

if TYPE_CHECKING:
    from fantasy_baseball_manager.valuation.models import PlayerValue


class ReplacementCalculator(Protocol):
    def compute_slot_costs(
        self,
        available_pool: list[PlayerValue],
        num_teams: int,
        num_keeper_slots: int,
    ) -> tuple[SlotCost, ...]: ...


class DraftPoolReplacementCalculator:
    """Compute replacement values by simulating a greedy best-available draft."""

    def __init__(self, user_pick_position: int) -> None:
        self._user_pick_position = user_pick_position

    def compute_slot_costs(
        self,
        available_pool: list[PlayerValue],
        num_teams: int,
        num_keeper_slots: int,
    ) -> tuple[SlotCost, ...]:
        sorted_pool = sorted(available_pool, key=lambda pv: pv.total_value, reverse=True)

        snake = generate_snake_order(num_teams, num_keeper_slots)

        # user_pick_position is 1-based; snake uses 0-based indices
        user_index = self._user_pick_position - 1

        user_pick_indices = [i for i, team_idx in enumerate(snake) if team_idx == user_index]

        slot_costs: list[SlotCost] = []
        for slot_number, pick_index in enumerate(user_pick_indices, start=1):
            replacement_value = sorted_pool[pick_index].total_value if pick_index < len(sorted_pool) else 0.0
            slot_costs.append(SlotCost(slot_number=slot_number, replacement_value=replacement_value))

        return tuple(slot_costs)
