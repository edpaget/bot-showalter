"""Draft plan generation from mock draft simulation results."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    AvailabilityWindow,
    DraftPlan,
    DraftPlanTarget,
    PickAvailability,
    PlayerAvailabilityCurve,
)

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


def _user_pick_numbers(slot: int, teams: int, total_rounds: int) -> list[int]:
    """Compute pick numbers for a user at *slot* (0-indexed) in a snake draft."""
    picks: list[int] = []
    for rnd in range(1, total_rounds + 1):
        pick = (rnd - 1) * teams + slot + 1 if rnd % 2 == 1 else rnd * teams - slot
        picks.append(pick)
    return picks


def compute_availability_windows(
    all_player_picks: dict[int, list[int]],
    player_names: dict[int, str],
    player_positions: dict[int, str],
    *,
    n_simulations: int,
    user_next_pick: int,
) -> list[AvailabilityWindow]:
    """Compute availability windows for all players in *all_player_picks*.

    For each player, computes percentile distribution and probability
    of being available at *user_next_pick*.
    """
    windows: list[AvailabilityWindow] = []
    for player_id, pick_numbers in all_player_picks.items():
        sorted_picks = sorted(pick_numbers)
        n_drafted = len(sorted_picks)

        # Percentiles (statistics.quantiles needs ≥ 2 data points)
        if n_drafted >= 2:
            quantile_values = statistics.quantiles(sorted_picks, n=20)
            earliest = quantile_values[0]  # 5th percentile
            latest = quantile_values[18]  # 95th percentile
        else:
            earliest = float(sorted_picks[0])
            latest = float(sorted_picks[0])

        median = float(statistics.median(sorted_picks))

        # Count of sims where this player was drafted before user_next_pick.
        # Undrafted sims (n_simulations - n_drafted) are implicitly counted as
        # available since they don't appear in drafted_before.
        drafted_before = sum(1 for p in sorted_picks if p < user_next_pick)
        available = (n_simulations - drafted_before) / n_simulations if n_simulations > 0 else 0.0

        windows.append(
            AvailabilityWindow(
                player_id=player_id,
                player_name=player_names.get(player_id, f"Unknown ({player_id})"),
                position=player_positions.get(player_id, "?"),
                earliest_pick=earliest,
                median_pick=median,
                latest_pick=latest,
                available_at_user_pick=available,
            )
        )

    return sorted(windows, key=lambda w: w.median_pick)


def compute_player_availability_curve(
    player_id: int,
    all_player_picks: dict[int, list[int]],
    player_names: dict[int, str],
    player_positions: dict[int, str],
    *,
    n_simulations: int,
    slot: int,
    teams: int,
    total_rounds: int,
) -> PlayerAvailabilityCurve:
    """Compute round-by-round availability curve for a single player."""
    user_picks = _user_pick_numbers(slot, teams, total_rounds)
    pick_numbers = all_player_picks.get(player_id, [])
    sorted_picks = sorted(pick_numbers)

    availabilities: list[PickAvailability] = []
    for rnd, pick in enumerate(user_picks, start=1):
        drafted_before = sum(1 for p in sorted_picks if p < pick)
        probability = (n_simulations - drafted_before) / n_simulations if n_simulations > 0 else 0.0
        availabilities.append(PickAvailability(round=rnd, pick=pick, probability=probability))

    return PlayerAvailabilityCurve(
        player_id=player_id,
        player_name=player_names.get(player_id, f"Unknown ({player_id})"),
        position=player_positions.get(player_id, "?"),
        pick_availabilities=availabilities,
    )
