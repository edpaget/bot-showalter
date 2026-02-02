from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

from fantasy_baseball_manager.draft.simulation_models import (
    SimulationConfig,
    SimulationPick,
    SimulationResult,
    TeamConfig,
    TeamResult,
)
from fantasy_baseball_manager.draft.state import DraftState

if TYPE_CHECKING:
    from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory


def generate_snake_order(num_teams: int, num_rounds: int) -> list[int]:
    """Return team indices in snake order: 0..n-1, n-1..0, ..."""
    order: list[int] = []
    for round_num in range(num_rounds):
        indices = list(range(num_teams))
        if round_num % 2 == 1:
            indices.reverse()
        order.extend(indices)
    return order


def simulate_draft(
    config: SimulationConfig,
    player_values: list[PlayerValue],
    player_positions: dict[tuple[str, str], tuple[str, ...]],
) -> SimulationResult:
    """Run a full simulated snake draft."""
    rng = random.Random(config.seed)

    # Build per-team DraftState instances
    team_states: dict[int, DraftState] = {}
    team_configs: dict[int, TeamConfig] = {}
    team_picks: dict[int, list[SimulationPick]] = defaultdict(list)

    for tc in config.teams:
        team_states[tc.team_id] = DraftState(
            roster_config=config.roster_config,
            player_values=player_values,
            player_positions=player_positions,
            category_weights=tc.strategy.category_weights,
        )
        team_configs[tc.team_id] = tc

    # Process keepers
    keeper_picks: list[SimulationPick] = []
    for tc in config.teams:
        for player_id in tc.keepers:
            # Find player name
            player_name = ""
            for pv in player_values:
                if pv.player_id == player_id:
                    player_name = pv.name
                    break

            # Draft as user in owning team, opponent in others
            position: str | None = None
            for tid, state in team_states.items():
                if tid == tc.team_id:
                    pick = state.draft_player(player_id, is_user=True)
                    position = pick.position
                else:
                    state.draft_player(player_id, is_user=False)

            sim_pick = SimulationPick(
                overall_pick=0,
                round_number=0,
                pick_in_round=0,
                team_id=tc.team_id,
                team_name=tc.name,
                player_id=player_id,
                player_name=player_name,
                position=position,
                adjusted_value=0.0,
            )
            keeper_picks.append(sim_pick)
            team_picks[tc.team_id].append(sim_pick)

    # Generate snake order
    team_list = [tc.team_id for tc in config.teams]
    num_teams = len(team_list)
    snake = generate_snake_order(num_teams, config.total_rounds)

    # Track globally drafted player IDs
    drafted: set[str] = {p.player_id for p in keeper_picks}

    pick_log: list[SimulationPick] = list(keeper_picks)
    overall_pick = 1

    for pick_index, team_idx in enumerate(snake):
        team_id = team_list[team_idx]
        tc = team_configs[team_id]
        state = team_states[team_id]
        strategy = tc.strategy

        round_number = pick_index // num_teams + 1
        pick_in_round = pick_index % num_teams + 1

        # Get rankings from this team's DraftState
        rankings = state.get_rankings(limit=50)

        # Filter out globally drafted players and apply rules
        best_candidate = None
        best_score = float("-inf")

        for ranking in rankings:
            if ranking.player_id in drafted:
                continue

            # Apply rules
            rule_multiplier = 1.0
            for rule in strategy.rules:
                # Find the PlayerValue for this ranking
                pv_for_rule: PlayerValue | None = None
                for pv in player_values:
                    if pv.player_id == ranking.player_id:
                        pv_for_rule = pv
                        break
                if pv_for_rule is None:
                    continue

                result = rule.evaluate(
                    player=pv_for_rule,
                    eligible_positions=ranking.eligible_positions,
                    round_number=round_number,
                    total_rounds=config.total_rounds,
                    picks_so_far=team_picks[team_id],
                )
                rule_multiplier *= result
                if rule_multiplier == 0.0:
                    break

            if rule_multiplier == 0.0:
                continue

            adjusted = ranking.adjusted_value * rule_multiplier
            noise = 0.0
            if strategy.noise_scale > 0 and abs(adjusted) > 0:
                noise = rng.gauss(0, strategy.noise_scale * abs(adjusted))

            score = adjusted + noise
            if score > best_score:
                best_score = score
                best_candidate = ranking

        if best_candidate is None:
            continue

        # Draft the player
        position = best_candidate.best_position
        for tid, s in team_states.items():
            if tid == team_id:
                s.draft_player(best_candidate.player_id, is_user=True, position=position)
            else:
                s.draft_player(best_candidate.player_id, is_user=False)

        drafted.add(best_candidate.player_id)

        sim_pick = SimulationPick(
            overall_pick=overall_pick,
            round_number=round_number,
            pick_in_round=pick_in_round,
            team_id=team_id,
            team_name=tc.name,
            player_id=best_candidate.player_id,
            player_name=best_candidate.name,
            position=position,
            adjusted_value=best_candidate.adjusted_value,
        )
        pick_log.append(sim_pick)
        team_picks[team_id].append(sim_pick)
        overall_pick += 1

    # Build team results with category totals
    # Create a lookup for player values by player_id
    pv_lookup: dict[str, list[PlayerValue]] = defaultdict(list)
    for pv in player_values:
        pv_lookup[pv.player_id].append(pv)

    team_results: list[TeamResult] = []
    for tc in config.teams:
        picks = tuple(team_picks.get(tc.team_id, []))
        category_totals: dict[StatCategory, float] = defaultdict(float)
        for pick in picks:
            for pv in pv_lookup.get(pick.player_id, []):
                for cv in pv.category_values:
                    category_totals[cv.category] += cv.raw_stat

        team_results.append(
            TeamResult(
                team_id=tc.team_id,
                team_name=tc.name,
                strategy_name=tc.strategy.name,
                picks=picks,
                category_totals=dict(category_totals),
            )
        )

    return SimulationResult(
        pick_log=tuple(pick_log),
        team_results=tuple(team_results),
        config=config,
    )
