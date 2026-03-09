"""Mock draft simulation engine."""

from __future__ import annotations

import random
import statistics
from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.domain import (
    BatchSimulationResult,
    DraftPick,
    DraftResult,
    PlayerDraftFrequency,
    SimulationSummary,
    StrategyComparison,
)
from fantasy_baseball_manager.services.draft_state import build_draft_roster_slots

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fantasy_baseball_manager.domain import DraftBoard, DraftBoardRow, LeagueSettings


class DraftBot(Protocol):
    """Protocol for mock draft bot strategies."""

    def pick(
        self,
        available: list[DraftBoardRow],
        roster: list[DraftPick],
        league: LeagueSettings,
    ) -> int:
        """Return the player_id to draft from the available list."""
        ...


def _snake_team(pick_number: int, teams: int) -> int:
    """Return 0-indexed team for a 1-indexed pick number in snake order."""
    zero_based = pick_number - 1
    round_number = zero_based // teams
    position_in_round = zero_based % teams
    if round_number % 2 == 0:
        return position_in_round
    return teams - 1 - position_in_round


def _compute_needs(roster: list[DraftPick], slots: dict[str, int]) -> dict[str, int]:
    """Return unfilled slot counts for a team."""
    filled: dict[str, int] = {}
    for pick in roster:
        filled[pick.position] = filled.get(pick.position, 0) + 1
    needs: dict[str, int] = {}
    for pos, total in slots.items():
        remaining = total - filled.get(pos, 0)
        if remaining > 0:
            needs[pos] = remaining
    return needs


# Composite slot eligibility: position -> list of composite slots it can fill
_COMPOSITE_SLOTS: dict[str, list[str]] = {
    "2B": ["MI"],
    "SS": ["MI"],
    "1B": ["CI"],
    "3B": ["CI"],
}


def _assign_position(player: DraftBoardRow, needs: dict[str, int]) -> str | None:
    """Assign a roster slot for the player based on needs.

    Returns the position string if assignable, None otherwise.
    Priority: primary position > composite slots (MI/CI) > flex (UTIL/P).
    """
    pos = player.position

    # Check primary position
    if pos in needs and needs[pos] > 0:
        return pos

    # Check composite slots (MI, CI)
    if player.player_type == "B":
        for composite in _COMPOSITE_SLOTS.get(pos, []):
            if composite in needs and needs[composite] > 0:
                return composite
        # Flex: UTIL for batters
        if "UTIL" in needs and needs["UTIL"] > 0:
            return "UTIL"
    elif player.player_type == "P":
        if "P" in needs and needs["P"] > 0:
            return "P"

    return None


def _available_for_team(
    pool: list[DraftBoardRow],
    needs: dict[str, int],
) -> list[DraftBoardRow]:
    """Filter pool to players assignable to this team, sorted by value descending."""
    assignable = [p for p in pool if _assign_position(p, needs) is not None]
    return sorted(assignable, key=lambda p: p.value, reverse=True)


def run_mock_draft(
    board: DraftBoard,
    league: LeagueSettings,
    strategies: Sequence[DraftBot],
    *,
    snake: bool = True,
    seed: int | None = None,
) -> DraftResult:
    """Run a complete mock draft simulation.

    Args:
        board: The draft board with ranked players.
        league: League settings defining roster slots.
        strategies: One bot per team (len must equal number of teams).
        snake: Use snake draft ordering (default True).
        seed: Random seed for reproducibility (reserved for future use;
              bots should be seeded externally).

    Returns:
        DraftResult with all picks and rosters.
    """
    _ = seed  # reserved for future use

    slots = build_draft_roster_slots(league)
    num_teams = len(strategies)
    total_rounds = sum(slots.values())
    total_picks = num_teams * total_rounds

    # Build mutable pool sorted by value descending
    pool = sorted(board.rows, key=lambda r: r.value, reverse=True)
    pool_set = {r.player_id for r in pool}

    rosters: dict[int, list[DraftPick]] = {i: [] for i in range(num_teams)}
    picks: list[DraftPick] = []

    for pick_num in range(1, total_picks + 1):
        team_idx = _snake_team(pick_num, num_teams) if snake else (pick_num - 1) % num_teams

        round_num = (pick_num - 1) // num_teams + 1
        needs = _compute_needs(rosters[team_idx], slots)
        available = _available_for_team(pool, needs)

        if not available:
            msg = f"No assignable players for team {team_idx} at pick {pick_num} (round {round_num}). Needs: {needs}"
            raise RuntimeError(msg)

        # Let the bot choose
        bot = strategies[team_idx]
        player_id = bot.pick(available, rosters[team_idx], league)

        # Find the chosen player in pool
        chosen: DraftBoardRow | None = None
        for row in pool:
            if row.player_id == player_id:
                chosen = row
                break

        if chosen is None or player_id not in pool_set:
            # Bot picked an invalid player — fall back to first available
            chosen = available[0]

        position = _assign_position(chosen, needs)
        if position is None:
            # Shouldn't happen since we filtered, but safety fallback
            chosen = available[0]
            position = _assign_position(chosen, needs)
            if position is None:  # pragma: no cover
                msg = f"Cannot assign position for {chosen.player_name}"
                raise RuntimeError(msg)

        draft_pick = DraftPick(
            round=round_num,
            pick=pick_num,
            team_idx=team_idx,
            player_id=chosen.player_id,
            player_name=chosen.player_name,
            position=position,
            value=chosen.value,
        )

        picks.append(draft_pick)
        rosters[team_idx].append(draft_pick)

        # Remove from pool
        pool = [r for r in pool if r.player_id != chosen.player_id]
        pool_set.discard(chosen.player_id)

    return DraftResult(picks=picks, rosters=rosters, snake=snake)


def run_batch_simulation(
    n_simulations: int,
    board: DraftBoard,
    league: LeagueSettings,
    user_strategy_factory: Callable[[random.Random], DraftBot],
    opponent_strategy_factories: Sequence[Callable[[random.Random], DraftBot]],
    *,
    draft_position: int | None = None,
    seed: int | None = None,
) -> BatchSimulationResult:
    """Run N mock drafts and aggregate results into analytics.

    Factories accept an RNG and return a fresh bot per simulation, enabling
    independent per-simulation seeding for reproducibility.
    ``len(opponent_strategy_factories)`` must equal ``league.teams - 1``.
    """
    num_teams = league.teams
    expected_opponents = num_teams - 1
    if len(opponent_strategy_factories) != expected_opponents:
        msg = f"Expected {expected_opponents} opponent factories, got {len(opponent_strategy_factories)}"
        raise ValueError(msg)

    user_values: list[float] = []
    user_rosters_all: list[list[DraftPick]] = []
    user_idx_per_sim: list[int] = []
    team_values_per_sim: list[list[float]] = []
    # player_id -> list of (round, pick) tuples when user drafted them
    player_drafts: dict[int, list[tuple[int, int]]] = {}
    # player_id -> player_name
    player_names: dict[int, str] = {r.player_id: r.player_name for r in board.rows}

    for i in range(n_simulations):
        sim_rng = random.Random(seed + i) if seed is not None else random.Random()  # noqa: S311

        # Determine user team_idx
        user_idx = draft_position if draft_position is not None else sim_rng.randrange(num_teams)
        user_idx_per_sim.append(user_idx)

        # Create bots from factories with independent RNGs
        user_bot = user_strategy_factory(random.Random(sim_rng.randint(0, 2**32)))  # noqa: S311
        strategies: list[DraftBot] = []
        opp_idx = 0
        for team_i in range(num_teams):
            if team_i == user_idx:
                strategies.append(user_bot)
            else:
                opp_rng = random.Random(sim_rng.randint(0, 2**32))  # noqa: S311
                strategies.append(opponent_strategy_factories[opp_idx](opp_rng))
                opp_idx += 1

        result = run_mock_draft(board, league, strategies)

        # Collect user roster value
        user_roster = result.rosters[user_idx]
        user_total = sum(p.value for p in user_roster)
        user_values.append(user_total)
        user_rosters_all.append(list(user_roster))

        # Collect per-team total values
        team_totals = [sum(p.value for p in result.rosters[t]) for t in range(num_teams)]
        team_values_per_sim.append(team_totals)

        # Track player draft info for user
        for pick in user_roster:
            if pick.player_id not in player_drafts:
                player_drafts[pick.player_id] = []
            player_drafts[pick.player_id].append((pick.round, pick.pick))

    # --- Aggregation ---

    # SimulationSummary
    deciles = statistics.quantiles(user_values, n=10)
    quartiles = statistics.quantiles(user_values, n=4)
    summary = SimulationSummary(
        n_simulations=n_simulations,
        team_idx=draft_position,
        avg_roster_value=statistics.mean(user_values),
        median_roster_value=statistics.median(user_values),
        p10_roster_value=deciles[0],
        p25_roster_value=quartiles[0],
        p75_roster_value=quartiles[2],
        p90_roster_value=deciles[8],
    )

    # PlayerDraftFrequency
    frequencies: list[PlayerDraftFrequency] = []
    for row in board.rows:
        pid = row.player_id
        drafts = player_drafts.get(pid, [])
        pct = len(drafts) / n_simulations
        if drafts:
            avg_round = statistics.mean(r for r, _ in drafts)
            avg_pick = statistics.mean(p for _, p in drafts)
        else:
            avg_round = 0.0
            avg_pick = 0.0
        frequencies.append(
            PlayerDraftFrequency(
                player_id=pid,
                player_name=player_names[pid],
                pct_drafted=pct,
                avg_round_drafted=avg_round,
                avg_pick_drafted=avg_pick,
            )
        )

    # StrategyComparison
    strategy_total_values: dict[str, list[float]] = {"user": []}
    for j in range(expected_opponents):
        strategy_total_values[f"opponent_{j}"] = []
    strategy_wins: dict[str, float] = {name: 0.0 for name in strategy_total_values}

    for i in range(n_simulations):
        user_idx = user_idx_per_sim[i]
        team_totals = team_values_per_sim[i]
        max_value = max(team_totals)

        # Map team indices to strategy names
        team_to_name: dict[int, str] = {}
        opp_counter = 0
        for t in range(num_teams):
            if t == user_idx:
                team_to_name[t] = "user"
            else:
                team_to_name[t] = f"opponent_{opp_counter}"
                opp_counter += 1

        # Count winners (ties split equally)
        winners = [t for t in range(num_teams) if team_totals[t] == max_value]
        share = 1.0 / len(winners)

        for t in range(num_teams):
            name = team_to_name[t]
            strategy_total_values[name].append(team_totals[t])
            if t in winners:
                strategy_wins[name] += share

    comparisons: list[StrategyComparison] = []
    for name in strategy_total_values:
        comparisons.append(
            StrategyComparison(
                strategy_name=name,
                avg_value=statistics.mean(strategy_total_values[name]),
                win_rate=strategy_wins[name] / n_simulations,
            )
        )

    return BatchSimulationResult(
        summary=summary,
        player_frequencies=frequencies,
        strategy_comparisons=comparisons,
        user_rosters=user_rosters_all,
        user_roster_values=user_values,
    )
