from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import BudgetAllocation

if TYPE_CHECKING:
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
