from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import BudgetAllocation, RoundTarget, SnakeDraftPlan

if TYPE_CHECKING:
    from collections.abc import Set

    from fantasy_baseball_manager.domain import LeagueSettings, PlayerTier, Valuation


def _build_slot_list(league: LeagueSettings) -> list[str]:
    """Expand league position config into a flat list of roster slots."""
    slots: list[str] = []
    for pos, count in league.positions.items():
        slots.extend([pos] * count)
    if league.roster_util > 0:
        slots.extend(["util"] * league.roster_util)
    for pos, count in league.pitcher_positions.items():
        slots.extend([pos] * count)
    return slots


def _group_by_position(valuations: list[Valuation]) -> dict[str, list[Valuation]]:
    """Group valuations by position, sorted descending by value."""
    groups: dict[str, list[Valuation]] = {}
    for v in valuations:
        groups.setdefault(v.position, []).append(v)
    for vals in groups.values():
        vals.sort(key=lambda v: v.value, reverse=True)
    return groups


def _find_target_players(
    position: str,
    budget: float,
    by_position: dict[str, list[Valuation]],
    player_names: dict[int, str],
) -> tuple[str, ...]:
    """Find players whose value is <= budget for the given position."""
    candidates = by_position.get(position, [])
    if position == "util":
        candidates = []
        for vals in by_position.values():
            candidates.extend(vals)
        candidates.sort(key=lambda v: v.value, reverse=True)

    matching = [v for v in candidates if v.value <= budget]
    if not matching and candidates:
        matching = [candidates[-1]]
    return tuple(player_names.get(v.player_id, f"Player {v.player_id}") for v in matching[:3])


def _find_target_tier(
    position: str,
    budget: float,
    tier_lookup: dict[str, list[PlayerTier]],
) -> int | None:
    """Find the best tier achievable at the given budget."""
    tiers = tier_lookup.get(position, [])
    for t in tiers:
        if t.value <= budget:
            return t.tier
    return None


def _balanced_allocation(
    slots: list[str],
    remaining: float,
    by_position: dict[str, list[Valuation]],
) -> dict[int, float]:
    """Distribute remaining budget proportional to position value pools."""
    slot_values: list[float] = []
    position_slot_index: dict[str, int] = {}
    for pos in slots:
        idx = position_slot_index.get(pos, 0)
        position_slot_index[pos] = idx + 1
        candidates = by_position.get(pos, [])
        if pos == "util":
            all_batters = []
            for vals in by_position.values():
                all_batters.extend(vals)
            all_batters.sort(key=lambda v: v.value, reverse=True)
            candidates = all_batters
        val = candidates[idx].value if idx < len(candidates) else 0.0
        slot_values.append(max(val, 0.0))

    total_value = sum(slot_values)
    allocations: dict[int, float] = {}
    if total_value == 0:
        return allocations

    n = len(slots)
    even_share = remaining / n if n > 0 else 0.0
    for i in range(n):
        proportional = (slot_values[i] / total_value) * remaining
        # Blend 50/50 between proportional and even to keep within 2x
        allocations[i] = 0.5 * proportional + 0.5 * even_share

    return allocations


def _stars_and_scrubs_allocation(
    slots: list[str],
    remaining: float,
    by_position: dict[str, list[Valuation]],
) -> dict[int, float]:
    """Greedily allocate budget to highest marginal-value slots."""
    slot_marginal: list[tuple[int, float]] = []
    position_slot_index: dict[str, int] = {}
    for i, pos in enumerate(slots):
        idx = position_slot_index.get(pos, 0)
        position_slot_index[pos] = idx + 1
        candidates = by_position.get(pos, [])
        if pos == "util":
            all_batters = []
            for vals in by_position.values():
                all_batters.extend(vals)
            all_batters.sort(key=lambda v: v.value, reverse=True)
            candidates = all_batters
        val = candidates[idx].value if idx < len(candidates) else 0.0
        slot_marginal.append((i, max(val, 0.0)))

    slot_marginal.sort(key=lambda x: x[1], reverse=True)
    allocations: dict[int, float] = {}
    budget_left = remaining

    for slot_idx, marginal_val in slot_marginal:
        if budget_left <= 0:
            allocations[slot_idx] = 0.0
            continue
        alloc = min(marginal_val, budget_left)
        allocations[slot_idx] = alloc
        budget_left -= alloc

    return allocations


def optimize_auction_budget(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
    strategy: str = "balanced",
    tiers: list[PlayerTier] | None = None,
) -> list[BudgetAllocation]:
    """Compute optimal auction budget allocation across roster positions."""
    slots = _build_slot_list(league)
    total_slots = len(slots)
    remaining = league.budget - total_slots

    if remaining < 0:
        remaining = 0

    by_position = _group_by_position(valuations)

    tier_lookup: dict[str, list[PlayerTier]] = {}
    if tiers is not None:
        for t in tiers:
            tier_lookup.setdefault(t.position, []).append(t)
        for tier_list in tier_lookup.values():
            tier_list.sort(key=lambda t: t.value, reverse=True)

    if strategy == "stars_and_scrubs":
        extra = _stars_and_scrubs_allocation(slots, remaining, by_position)
    else:
        extra = _balanced_allocation(slots, remaining, by_position)

    result: list[BudgetAllocation] = []
    for i, pos in enumerate(slots):
        budget = 1.0 + extra.get(i, 0.0)
        target_tier = _find_target_tier(pos, budget, tier_lookup) if tiers is not None else None
        target_names = _find_target_players(pos, budget, by_position, player_names)
        result.append(
            BudgetAllocation(
                position=pos,
                budget=budget,
                target_tier=target_tier,
                target_player_names=target_names,
            )
        )

    # Fix floating point: ensure exact budget sum
    current_total = sum(a.budget for a in result)
    if result and current_total != league.budget:
        diff = league.budget - current_total
        max_idx = max(range(len(result)), key=lambda idx: result[idx].budget)
        old = result[max_idx]
        result[max_idx] = BudgetAllocation(
            position=old.position,
            budget=old.budget + diff,
            target_tier=old.target_tier,
            target_player_names=old.target_player_names,
        )

    return result


_PITCHER_TYPES = frozenset({"sp", "rp"})
_BATTER_TYPES = frozenset({"c", "1b", "2b", "3b", "ss", "of", "dh"})


def _snake_pick_numbers(draft_slot: int, teams: int, total_rounds: int) -> list[int]:
    """Return 1-indexed overall pick numbers for *draft_slot* in a snake draft."""
    picks: list[int] = []
    for rnd in range(total_rounds):
        pick = rnd * teams + draft_slot if rnd % 2 == 0 else rnd * teams + (teams - draft_slot + 1)
        picks.append(pick)
    return picks


def _build_needs(
    slots: list[str],
    my_keepers: list[tuple[int, str]] | None,
) -> dict[str, int]:
    """Build a dict of {position: remaining_slots} after accounting for keepers."""
    needs: dict[str, int] = {}
    for pos in slots:
        needs[pos] = needs.get(pos, 0) + 1
    if my_keepers:
        for _pid, pos in my_keepers:
            lpos = pos.lower()
            if lpos in needs and needs[lpos] > 0:
                needs[lpos] -= 1
            elif "util" in needs and needs["util"] > 0 and lpos in _BATTER_TYPES:
                needs["util"] -= 1
            elif "p" in needs and needs["p"] > 0 and lpos in _PITCHER_TYPES:
                needs["p"] -= 1
    return {pos: count for pos, count in needs.items() if count > 0}


def _positions_for_slot(slot_pos: str) -> Set[str]:
    """Return the set of valuation positions that can fill *slot_pos*."""
    if slot_pos == "util":
        return _BATTER_TYPES
    if slot_pos == "p":
        return _PITCHER_TYPES
    return frozenset({slot_pos})


def _best_available(
    pool: dict[str, list[Valuation]],
    candidate_positions: Set[str],
) -> Valuation | None:
    """Return the highest-value available player matching any of *candidate_positions*."""
    best: Valuation | None = None
    for pos in candidate_positions:
        candidates = pool.get(pos)
        if candidates and (best is None or candidates[0].value > best.value):
            best = candidates[0]
    return best


def _remove_from_pool(pool: dict[str, list[Valuation]], player_id: int) -> None:
    """Remove a player from every position list in the pool."""
    for candidates in pool.values():
        for i, v in enumerate(candidates):
            if v.player_id == player_id:
                candidates.pop(i)
                break


def _simulate_opponent_picks(
    pool: dict[str, list[Valuation]],
    n_picks: int,
) -> None:
    """Remove *n_picks* best-available players from the pool (opponent simulation)."""
    for _ in range(n_picks):
        best: Valuation | None = None
        for candidates in pool.values():
            if candidates and (best is None or candidates[0].value > best.value):
                best = candidates[0]
        if best is not None:
            _remove_from_pool(pool, best.player_id)


def plan_snake_draft(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
    draft_slot: int,
    tiers: list[PlayerTier] | None = None,
    my_keepers: list[tuple[int, str]] | None = None,
    league_keeper_ids: set[int] | None = None,
    keepers_per_team: int = 0,
) -> SnakeDraftPlan:
    """Produce a round-by-round snake draft plan for the given draft slot."""
    slots = _build_slot_list(league)
    total_slots = len(slots)
    total_rounds = total_slots - keepers_per_team

    # Build needs (positions still to fill)
    needs = _build_needs(slots, my_keepers)

    # Build pool — remove league keepers
    pool_valuations = valuations
    if league_keeper_ids:
        pool_valuations = [v for v in pool_valuations if v.player_id not in league_keeper_ids]
    pool = _group_by_position(pool_valuations)

    # Build tier lookup
    tier_lookup: dict[int, int] = {}
    if tiers is not None:
        for t in tiers:
            tier_lookup[t.player_id] = t.tier

    # Calculate pick numbers
    pick_numbers = _snake_pick_numbers(draft_slot, league.teams, total_rounds)

    rounds: list[RoundTarget] = []

    for round_idx, pick_number in enumerate(pick_numbers):
        # Simulate opponent picks between last user pick and this one
        opponent_picks = pick_number - 1 if round_idx == 0 else abs(pick_number - pick_numbers[round_idx - 1]) - 1
        _simulate_opponent_picks(pool, opponent_picks)

        # For each unfilled position slot, compute marginal value
        scored: list[tuple[str, float, Valuation]] = []
        for slot_pos, remaining in needs.items():
            if remaining <= 0:
                continue
            candidate_positions = _positions_for_slot(slot_pos)
            best = _best_available(pool, candidate_positions)
            if best is None:
                continue

            # Replacement value: value expected at NEXT pick
            if round_idx + 1 < len(pick_numbers):
                next_pick = pick_numbers[round_idx + 1]
                picks_between = abs(next_pick - pick_number) - 1
                # Estimate how many of this position will be taken
                # by counting down the pool
                candidates = pool.get(best.position, [])
                # After picks_between opponent picks, roughly
                # (picks_between / total_remaining_positions) * len(candidates) taken
                # Simplified: look at the (picks_between // league.teams + 1)th player
                depth = picks_between // league.teams + 1
                replacement_value = candidates[depth].value if depth < len(candidates) else 0.0
            else:
                replacement_value = 0.0

            marginal = best.value - replacement_value
            scored.append((slot_pos, marginal, best))

        if not scored:
            break

        # Sort by marginal value descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best_slot, _best_marginal, best_player = scored[0]

        # Alternatives: next best positions
        alternatives = tuple(s[0] for s in scored[1:4])

        tier_val = tier_lookup.get(best_player.player_id)

        rounds.append(
            RoundTarget(
                round=round_idx + 1,
                pick_number=pick_number,
                recommended_position=best_slot,
                target_tier=tier_val,
                expected_value=best_player.value,
                alternative_positions=alternatives,
            )
        )

        # Update state
        _remove_from_pool(pool, best_player.player_id)
        needs[best_slot] -= 1
        if needs[best_slot] <= 0:
            del needs[best_slot]

    return SnakeDraftPlan(
        draft_slot=draft_slot,
        teams=league.teams,
        rounds=rounds,
    )
