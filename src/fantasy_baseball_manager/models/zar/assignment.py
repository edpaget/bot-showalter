from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssignmentResult:
    """Result of optimal position assignment."""

    assignments: dict[int, str]  # player_index → assigned position
    replacement: dict[str, float]  # position → replacement level
    var_values: list[float]  # per-player VAR (0.0 for unassigned)


def assign_positions(
    composite_scores: list[float],
    player_positions: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
) -> AssignmentResult:
    """Optimally assign players to position slots using the Hungarian algorithm.

    Maximizes total composite score across all assignments while respecting
    slot capacities and position eligibility.
    """
    if not composite_scores:
        return AssignmentResult(assignments={}, replacement={}, var_values=[])

    # Step 1: Expand slots into flat list
    slots: list[str] = []
    for position, count in roster_spots.items():
        slots.extend([position] * (count * num_teams))
    total_slots = len(slots)

    if total_slots == 0:
        return AssignmentResult(
            assignments={},
            replacement={pos: 0.0 for pos in roster_spots},
            var_values=[0.0] * len(composite_scores),
        )

    # Step 2: Select candidates — top min(len, 1.5x slots) by score descending
    n_candidates = min(len(composite_scores), int(total_slots * 1.5))
    indexed_scores = sorted(enumerate(composite_scores), key=lambda x: x[1], reverse=True)
    candidate_indices = [idx for idx, _ in indexed_scores[:n_candidates]]

    # Step 3: Build cost matrix (candidates + dummies × slots).
    # Dummy rows ensure feasibility when fewer candidates are eligible than slots.
    # Real candidates: cost = -score if eligible, else PENALTY.
    # Dummy rows: cost = PENALTY for all slots.
    penalty = 1e18
    n_rows = max(n_candidates, total_slots)
    cost = np.full((n_rows, total_slots), penalty)
    for row, orig_idx in enumerate(candidate_indices):
        eligible = set(player_positions[orig_idx])
        score = composite_scores[orig_idx]
        for col, slot_pos in enumerate(slots):
            if slot_pos in eligible:
                cost[row, col] = -score

    # Step 4: Run the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost)

    # Step 5: Map back to original indices and positions (skip dummy/penalty assignments)
    assignments: dict[int, str] = {}
    for r, c in zip(row_ind, col_ind, strict=True):
        if r < n_candidates and cost[r, c] < penalty:
            orig_idx = candidate_indices[r]
            assignments[orig_idx] = slots[c]

    # Step 6: Derive per-position replacement levels
    replacement: dict[str, float] = {}
    for position in roster_spots:
        assigned_at_pos = [composite_scores[i] for i, pos in assignments.items() if pos == position]
        if assigned_at_pos:
            replacement[position] = min(assigned_at_pos)
        else:
            replacement[position] = 0.0

    # Step 7: Compute VAR
    var_values = [0.0] * len(composite_scores)
    for idx, pos in assignments.items():
        var_values[idx] = composite_scores[idx] - replacement[pos]

    return AssignmentResult(
        assignments=assignments,
        replacement=replacement,
        var_values=var_values,
    )
