"""Draft plan generation from mock draft simulation results."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import DraftPlan, DraftPlanTarget

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftPick


def generate_draft_plan(
    user_rosters: list[list[DraftPick]],
    *,
    slot: int,
    teams: int,
    strategy_name: str,
) -> DraftPlan:
    """Distill per-round position-targeting patterns from simulation rosters.

    For each round, finds the modal position drafted across simulations,
    computes confidence (fraction of sims that drafted that position),
    and merges adjacent rounds with the same modal position into ranges.
    """
    if not user_rosters:
        return DraftPlan(
            slot=slot,
            teams=teams,
            strategy_name=strategy_name,
            targets=[],
            n_simulations=0,
            avg_roster_value=0.0,
        )

    n_sims = len(user_rosters)

    # round -> position -> count
    round_position_counts: dict[int, Counter[str]] = {}
    # round -> position -> player_name -> count
    round_position_players: dict[int, dict[str, Counter[str]]] = {}

    for roster in user_rosters:
        for pick in roster:
            rnd = pick.round

            if rnd not in round_position_counts:
                round_position_counts[rnd] = Counter()
            round_position_counts[rnd][pick.position] += 1

            if rnd not in round_position_players:
                round_position_players[rnd] = {}
            if pick.position not in round_position_players[rnd]:
                round_position_players[rnd][pick.position] = Counter()
            round_position_players[rnd][pick.position][pick.player_name] += 1

    # For each round, find modal position and confidence
    rounds_sorted = sorted(round_position_counts.keys())
    round_modal: list[tuple[int, str, float]] = []  # (round, position, confidence)

    for rnd in rounds_sorted:
        counts = round_position_counts[rnd]
        modal_pos = counts.most_common(1)[0][0]
        confidence = counts[modal_pos] / n_sims
        round_modal.append((rnd, modal_pos, confidence))

    # Merge adjacent rounds with same modal position
    targets: list[DraftPlanTarget] = []
    i = 0
    while i < len(round_modal):
        start_round, pos, _ = round_modal[i]
        end_round = start_round
        j = i + 1
        while j < len(round_modal) and round_modal[j][1] == pos:
            end_round = round_modal[j][0]
            j += 1

        # Compute merged confidence: total modal-position picks / total picks across range
        total_modal = 0
        total_sims_in_range = 0
        merged_player_counts: Counter[str] = Counter()
        for k in range(i, j):
            rnd_k = round_modal[k][0]
            total_modal += round_position_counts[rnd_k][pos]
            total_sims_in_range += n_sims
            if rnd_k in round_position_players and pos in round_position_players[rnd_k]:
                merged_player_counts += round_position_players[rnd_k][pos]

        confidence = total_modal / total_sims_in_range
        example_players = [name for name, _ in merged_player_counts.most_common(3)]

        targets.append(
            DraftPlanTarget(
                round_range=(start_round, end_round),
                position=pos,
                confidence=confidence,
                example_players=example_players,
            )
        )
        i = j

    avg_value = statistics.mean(sum(p.value for p in roster) for roster in user_rosters)

    return DraftPlan(
        slot=slot,
        teams=teams,
        strategy_name=strategy_name,
        targets=targets,
        n_simulations=n_sims,
        avg_roster_value=avg_value,
    )
