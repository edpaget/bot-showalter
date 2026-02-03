from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from fantasy_baseball_manager.keeper.models import (
    KeeperCandidate,
    KeeperRecommendation,
    KeeperSurplus,
    SlotCost,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.keeper.replacement import ReplacementCalculator
    from fantasy_baseball_manager.valuation.models import PlayerValue


class SurplusCalculator:
    """Compute keeper surplus values over draft replacement level."""

    def __init__(
        self,
        replacement_calculator: ReplacementCalculator,
        num_teams: int,
        num_keeper_slots: int,
    ) -> None:
        self._replacement_calculator = replacement_calculator
        self._num_teams = num_teams
        self._num_keeper_slots = num_keeper_slots

    def rank_candidates(
        self,
        candidates: list[KeeperCandidate],
        all_player_values: list[PlayerValue],
        other_keepers: set[str],
    ) -> list[KeeperSurplus]:
        """Rank candidates by surplus value using a single pool computation."""
        candidate_ids = {c.player_id for c in candidates}
        excluded = candidate_ids | other_keepers
        pool = [pv for pv in all_player_values if pv.player_id not in excluded]

        slot_costs = self._replacement_calculator.compute_slot_costs(
            pool, self._num_teams, self._num_keeper_slots
        )

        return self._assign_candidates_to_slots(candidates, slot_costs)

    def find_optimal_keepers(
        self,
        candidates: list[KeeperCandidate],
        all_player_values: list[PlayerValue],
        other_keepers: set[str],
    ) -> KeeperRecommendation:
        """Find the combination of keepers that maximizes total surplus."""
        best_surplus = float("-inf")
        best_combo: tuple[KeeperCandidate, ...] = ()
        best_assignments: list[KeeperSurplus] = []

        num_to_keep = min(self._num_keeper_slots, len(candidates))

        for combo in combinations(candidates, num_to_keep):
            combo_ids = {c.player_id for c in combo}
            excluded = combo_ids | other_keepers
            pool = [pv for pv in all_player_values if pv.player_id not in excluded]

            slot_costs = self._replacement_calculator.compute_slot_costs(
                pool, self._num_teams, self._num_keeper_slots
            )

            assignments = self._assign_candidates_to_slots(list(combo), slot_costs)
            total = sum(a.surplus_value for a in assignments)

            if total > best_surplus:
                best_surplus = total
                best_combo = combo
                best_assignments = assignments

        # Compute surplus for all candidates using the optimal combo's pool
        optimal_ids = {c.player_id for c in best_combo}
        excluded_for_all = optimal_ids | other_keepers
        pool_for_all = [pv for pv in all_player_values if pv.player_id not in excluded_for_all]
        slot_costs_for_all = self._replacement_calculator.compute_slot_costs(
            pool_for_all, self._num_teams, self._num_keeper_slots
        )
        all_assignments = self._assign_candidates_to_slots(candidates, slot_costs_for_all)

        return KeeperRecommendation(
            keepers=tuple(best_assignments),
            total_surplus=best_surplus,
            all_candidates=tuple(all_assignments),
        )

    @staticmethod
    def _assign_candidates_to_slots(
        candidates: list[KeeperCandidate],
        slot_costs: tuple[SlotCost, ...],
    ) -> list[KeeperSurplus]:
        """Assign candidates to slots: highest value player to highest cost slot."""
        sorted_candidates = sorted(
            candidates, key=lambda c: c.player_value.total_value, reverse=True
        )

        results: list[KeeperSurplus] = []
        for i, candidate in enumerate(sorted_candidates):
            if i < len(slot_costs):
                slot = slot_costs[i]
                surplus = candidate.player_value.total_value - slot.replacement_value
                results.append(
                    KeeperSurplus(
                        player_id=candidate.player_id,
                        name=candidate.name,
                        player_value=candidate.player_value.total_value,
                        eligible_positions=candidate.eligible_positions,
                        assigned_slot=slot.slot_number,
                        replacement_value=slot.replacement_value,
                        surplus_value=surplus,
                        category_values=candidate.player_value.category_values,
                    )
                )
            else:
                # More candidates than slots — assign to a virtual slot with 0 replacement
                results.append(
                    KeeperSurplus(
                        player_id=candidate.player_id,
                        name=candidate.name,
                        player_value=candidate.player_value.total_value,
                        eligible_positions=candidate.eligible_positions,
                        assigned_slot=len(slot_costs) + (i - len(slot_costs) + 1),
                        replacement_value=0.0,
                        surplus_value=candidate.player_value.total_value,
                        category_values=candidate.player_value.category_values,
                    )
                )

        return sorted(results, key=lambda s: s.surplus_value, reverse=True)
