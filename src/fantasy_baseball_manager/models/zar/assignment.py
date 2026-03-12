from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssignmentResult:
    """Result of optimal position assignment."""

    assignments: dict[int, str]  # player_index → assigned position
    replacement: dict[str, float]  # position → replacement level
    var_values: list[float]  # per-player VAR (0.0 for unassigned)


_FLEX_POSITIONS = frozenset({"UTIL", "P", "util", "p"})


def _normalize_flex_assignments(
    assignments: dict[int, str],
    composite_scores: list[float],
    player_positions: list[list[str]],
) -> None:
    """Swap flex-assigned players to specific positions when possible.

    The Hungarian solver maximizes total assigned score, but when a player
    qualifies for both a specific position and a flex slot (UTIL/P), the
    total score is the same either way.  This post-processing step pushes
    the *best* eligible players to specific positions and the *weakest* to
    flex, producing more natural labels (e.g. Bobby Witt → SS, not UTIL).
    """
    # Index assigned players by position
    by_position: dict[str, list[int]] = {}
    for idx, pos in assignments.items():
        by_position.setdefault(pos, []).append(idx)

    changed = True
    while changed:
        changed = False
        for flex_pos in _FLEX_POSITIONS:
            flex_players = by_position.get(flex_pos, [])
            # Highest-scoring flex players get first chance to move
            flex_players.sort(key=lambda i: composite_scores[i], reverse=True)

            for flex_idx in list(flex_players):
                eligible_specific = [p for p in player_positions[flex_idx] if p not in _FLEX_POSITIONS]
                for target_pos in eligible_specific:
                    target_players = by_position.get(target_pos, [])
                    if not target_players:
                        continue
                    weakest_idx = min(target_players, key=lambda i: composite_scores[i])
                    # Swap only if the flex player is better AND the displaced player can go flex
                    if composite_scores[flex_idx] > composite_scores[weakest_idx] and flex_pos in set(
                        player_positions[weakest_idx]
                    ):
                        assignments[flex_idx] = target_pos
                        assignments[weakest_idx] = flex_pos
                        by_position[flex_pos].remove(flex_idx)
                        by_position[flex_pos].append(weakest_idx)
                        by_position[target_pos].remove(weakest_idx)
                        by_position[target_pos].append(flex_idx)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break


def assign_positions(
    composite_scores: list[float],
    player_positions: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
) -> AssignmentResult:
    """Optimally assign players to position slots using the Hungarian algorithm.

    Maximizes total composite score across all assignments while respecting
    slot capacities and position eligibility.  After the solver runs, position
    labels are normalized so the best players at each position get their
    specific slot rather than a flex slot (UTIL/P).
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
    penalty = 1e9
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

    # Step 5b: Normalize — push best players to specific positions, weakest to flex
    _normalize_flex_assignments(assignments, composite_scores, player_positions)

    # Step 6: Derive per-position replacement levels.
    # Specific positions use the standard min-of-assigned computation.
    # Flex positions (P/UTIL) use inflated position counts: flex-overflow
    # players are counted toward their primary specific position, so the
    # replacement level reflects the full demand for that position (specific
    # slots + flex overflow).  This eliminates the value cliff between
    # the last specific-slot player and the first flex-slot player.

    # 6a: Compute raw specific-position replacement (for primary-pos tiebreak)
    raw_specific: dict[str, float] = {}
    for position in roster_spots:
        if position not in _FLEX_POSITIONS:
            scores_at = [composite_scores[i] for i, p in assignments.items() if p == position]
            raw_specific[position] = min(scores_at) if scores_at else 0.0

    # 6b: Assign each flex-assigned player a primary specific position
    # (the eligible specific position with the highest raw replacement —
    # i.e. the scarcest position they could fill).
    primary_pos: dict[int, str] = {}
    for idx, pos in assignments.items():
        if pos in _FLEX_POSITIONS:
            eligible_specific = [p for p in player_positions[idx] if p not in _FLEX_POSITIONS and p in raw_specific]
            if eligible_specific:
                primary_pos[idx] = max(eligible_specific, key=lambda p: raw_specific[p])

    # 6c: Compute inflated replacement levels — each specific position's
    # replacement is the worst assigned player who "counts" toward it
    # (either directly assigned or flex-overflow with it as primary).
    replacement: dict[str, float] = {}
    for position in roster_spots:
        if position in _FLEX_POSITIONS:
            scores_at = [composite_scores[i] for i, p in assignments.items() if p == position]
            replacement[position] = min(scores_at) if scores_at else 0.0
        else:
            scores_at = []
            for idx, apos in assignments.items():
                if apos == position or apos in _FLEX_POSITIONS and primary_pos.get(idx) == position:
                    scores_at.append(composite_scores[idx])
            replacement[position] = min(scores_at) if scores_at else 0.0

    # Step 7: Compute VAR.  Flex-assigned players with a primary specific
    # position use that position's (inflated) replacement level.
    var_values = [0.0] * len(composite_scores)
    for idx, pos in assignments.items():
        if pos in _FLEX_POSITIONS and idx in primary_pos:
            var_values[idx] = composite_scores[idx] - replacement[primary_pos[idx]]
        else:
            var_values[idx] = composite_scores[idx] - replacement[pos]

    return AssignmentResult(
        assignments=assignments,
        replacement=replacement,
        var_values=var_values,
    )
