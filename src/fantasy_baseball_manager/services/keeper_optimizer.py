import dataclasses
import heapq
import itertools
import math
from collections import Counter, defaultdict

from fantasy_baseball_manager.domain import (
    KeeperConstraints,
    KeeperDecision,
    KeeperSet,
    KeeperSolution,
    LeagueSettings,
    Player,
    SensitivityEntry,
    Valuation,
)


def solve_keepers(
    candidates: list[KeeperDecision],
    constraints: KeeperConstraints,
) -> KeeperSolution:
    """Find the optimal keeper set maximizing total surplus, subject to constraints."""
    candidate_ids = {c.player_id for c in candidates}
    for pid in constraints.required_keepers:
        if pid not in candidate_ids:
            msg = f"required keeper player_id={pid} not in candidates"
            raise ValueError(msg)

    valid_sets = _find_valid_sets(candidates, constraints)

    if not valid_sets:
        msg = "no valid keeper sets satisfy the given constraints"
        raise ValueError(msg)

    valid_sets.sort(key=lambda s: s.score, reverse=True)

    optimal = valid_sets[0]
    alternatives = valid_sets[1:6]
    sensitivity = _compute_sensitivity(optimal, valid_sets)

    return KeeperSolution(
        optimal=optimal,
        alternatives=alternatives,
        sensitivity=sensitivity,
    )


def _find_valid_sets(
    candidates: list[KeeperDecision],
    constraints: KeeperConstraints,
) -> list[KeeperSet]:
    required_ids = set(constraints.required_keepers)
    by_id = {c.player_id: c for c in candidates}
    required = [by_id[pid] for pid in constraints.required_keepers]
    optional = sorted(
        [c for c in candidates if c.player_id not in required_ids],
        key=lambda c: c.surplus,
        reverse=True,
    )

    k = constraints.max_keepers - len(required)
    if k < 0:
        msg = "more required keepers than max_keepers"
        raise ValueError(msg)

    # Check that required alone don't violate constraints
    if not _is_valid(tuple(required), constraints, partial=True):
        msg = "no valid keeper sets satisfy the given constraints"
        raise ValueError(msg)

    if len(optional) <= k:
        # Must take all optional players
        all_players = tuple(required) + tuple(optional)
        if _is_valid(all_players, constraints, partial=False):
            return [_build_keeper_set(all_players)]
        msg = "no valid keeper sets satisfy the given constraints"
        raise ValueError(msg)

    n = len(optional)
    if math.comb(n, k) <= 100_000:
        return _enumerate(required, optional, k, constraints)
    return _branch_and_bound(required, optional, k, constraints)


def _is_valid(
    players: tuple[KeeperDecision, ...],
    constraints: KeeperConstraints,
    *,
    partial: bool = False,
) -> bool:
    if constraints.max_per_position is not None:
        counts: Counter[str] = Counter()
        for p in players:
            counts[p.position] += 1
        for pos, limit in constraints.max_per_position.items():
            if counts[pos] > limit:
                return False

    if not partial and constraints.max_cost is not None:
        total_cost = sum(p.cost for p in players)
        if total_cost > constraints.max_cost:
            return False

    return True


def _build_keeper_set(players: tuple[KeeperDecision, ...]) -> KeeperSet:
    total_surplus = sum(p.surplus for p in players)
    total_cost = sum(p.cost for p in players)
    positions_filled: Counter[str] = Counter()
    for p in players:
        positions_filled[p.position] += 1
    return KeeperSet(
        players=players,
        total_surplus=total_surplus,
        total_cost=total_cost,
        positions_filled=dict(positions_filled),
        score=total_surplus,
    )


def _enumerate(
    required: list[KeeperDecision],
    optional: list[KeeperDecision],
    k: int,
    constraints: KeeperConstraints,
) -> list[KeeperSet]:
    results: list[KeeperSet] = []
    req_tuple = tuple(required)
    for combo in itertools.combinations(optional, k):
        players = req_tuple + combo
        if _is_valid(players, constraints):
            results.append(_build_keeper_set(players))
    return results


def _branch_and_bound(
    required: list[KeeperDecision],
    optional: list[KeeperDecision],
    k: int,
    constraints: KeeperConstraints,
    buffer_size: int = 50,
) -> list[KeeperSet]:
    """DFS with upper-bound pruning. Collects top buffer_size sets via min-heap."""
    # optional is already sorted by surplus desc
    req_tuple = tuple(required)
    req_cost = sum(p.cost for p in required)
    req_positions: Counter[str] = Counter(p.position for p in required)

    # Prefix sums for upper bound calculation
    n = len(optional)
    suffix_surplus = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_surplus[i] = suffix_surplus[i + 1] + optional[i].surplus

    # Min-heap of (score, KeeperSet) — we keep the top buffer_size
    heap: list[tuple[float, int, KeeperSet]] = []
    heap_counter = 0
    min_score = float("-inf")

    def _dfs(
        idx: int,
        chosen: list[KeeperDecision],
        current_surplus: float,
        current_cost: float,
        pos_counts: Counter[str],
        remaining: int,
    ) -> None:
        nonlocal heap_counter, min_score

        if remaining == 0:
            players = req_tuple + tuple(chosen)
            if constraints.max_cost is not None and current_cost > constraints.max_cost:
                return
            ks = _build_keeper_set(players)
            heap_counter += 1
            if len(heap) < buffer_size:
                heapq.heappush(heap, (ks.score, heap_counter, ks))
                if len(heap) == buffer_size:
                    min_score = heap[0][0]
            elif ks.score > min_score:
                heapq.heapreplace(heap, (ks.score, heap_counter, ks))
                min_score = heap[0][0]
            return

        if idx >= n:
            return

        # Not enough candidates left
        if n - idx < remaining:
            return

        # Upper bound: current surplus + best possible remaining
        # Take the top `remaining` surpluses from optional[idx:]
        # Since sorted desc, that's suffix_surplus[idx] minus suffix_surplus[idx + remaining]
        end = min(idx + remaining, n)
        upper_bound = current_surplus
        for i in range(idx, end):
            upper_bound += optional[i].surplus

        if len(heap) >= buffer_size and upper_bound <= min_score:
            return

        # Try including optional[idx]
        player = optional[idx]
        pos = player.position
        pos_counts[pos] += 1

        pos_ok = True
        if constraints.max_per_position is not None:
            limit = constraints.max_per_position.get(pos)
            if limit is not None and pos_counts[pos] > limit:
                pos_ok = False

        cost_ok = True
        new_cost = current_cost + player.cost
        if constraints.max_cost is not None and new_cost > constraints.max_cost and remaining == 1:
            cost_ok = False

        if pos_ok and cost_ok:
            chosen.append(player)
            _dfs(
                idx + 1,
                chosen,
                current_surplus + player.surplus,
                new_cost,
                pos_counts,
                remaining - 1,
            )
            chosen.pop()

        pos_counts[pos] -= 1

        # Try skipping optional[idx]
        _dfs(idx + 1, chosen, current_surplus, current_cost, pos_counts, remaining)

    req_surplus = sum(p.surplus for p in required)
    _dfs(0, [], req_surplus, req_cost, Counter(req_positions), k)

    return [item[2] for item in heap]


def _compute_sensitivity(
    optimal: KeeperSet,
    sorted_sets: list[KeeperSet],
) -> list[SensitivityEntry]:
    optimal_ids = frozenset(p.player_id for p in optimal.players)
    entries: list[SensitivityEntry] = []

    for player in optimal.players:
        gap = float("inf")
        for ks in sorted_sets:
            ks_ids = frozenset(p.player_id for p in ks.players)
            if ks_ids == optimal_ids:
                continue
            if player.player_id not in ks_ids:
                gap = optimal.score - ks.score
                break
        entries.append(
            SensitivityEntry(
                player_name=player.player_name,
                player_id=player.player_id,
                surplus_gap=gap,
            )
        )

    entries.sort(key=lambda e: e.surplus_gap)
    return entries


def compute_adjusted_draft_pool(
    league_keeper_ids: set[int],
    valuations: list[Valuation],
    league: LeagueSettings,
) -> tuple[list[Valuation], dict[str, float]]:
    """Remove kept players from the pool and compute replacement-level baselines."""
    filtered = [v for v in valuations if v.player_id not in league_keeper_ids]

    # Group by position (case-normalized)
    by_position: dict[str, list[Valuation]] = defaultdict(list)
    for v in filtered:
        by_position[v.position.lower()].append(v)
    for pos_list in by_position.values():
        pos_list.sort(key=lambda v: v.value, reverse=True)

    # Compute replacement level per position
    all_positions = {pos.lower(): slots for pos, slots in league.positions.items()}
    all_positions.update({pos.lower(): slots for pos, slots in league.pitcher_positions.items()})

    replacement_levels: dict[str, float] = {}
    for pos, slots in all_positions.items():
        pool = by_position.get(pos, [])
        rank = slots * league.teams
        if rank < len(pool):
            replacement_levels[pos] = pool[rank].value
        else:
            replacement_levels[pos] = 0.0

    filtered.sort(key=lambda v: v.value, reverse=True)
    return filtered, replacement_levels


def _estimated_draft_value(
    keeper_set: KeeperSet,
    base_pool: list[Valuation],
    league: LeagueSettings,
) -> float:
    """Estimate total value from drafting remaining roster slots."""
    keeper_ids = {p.player_id for p in keeper_set.players}
    available = [v for v in base_pool if v.player_id not in keeper_ids]

    # Compute unfilled slots per position
    all_positions = {pos.lower(): slots for pos, slots in league.positions.items()}
    all_positions.update({pos.lower(): slots for pos, slots in league.pitcher_positions.items()})

    unfilled: dict[str, int] = {}
    for pos, slots in all_positions.items():
        filled = keeper_set.positions_filled.get(pos, 0)
        unfilled[pos] = max(0, slots - filled)

    # Group available by position
    by_position: dict[str, list[float]] = defaultdict(list)
    for v in available:
        by_position[v.position.lower()].append(v.value)

    total_value = 0.0
    used_ids: set[int] = set()

    # Fill position-specific slots first
    for pos, slots_needed in unfilled.items():
        pos_values = by_position.get(pos, [])
        for i in range(min(slots_needed, len(pos_values))):
            total_value += pos_values[i]
            # Track which position-slot values we've consumed
            # Find the actual player to mark as used
        # Mark consumed players as used from available pool
        consumed = 0
        for v in available:
            if v.position.lower() == pos and v.player_id not in used_ids:
                used_ids.add(v.player_id)
                consumed += 1
                if consumed >= min(slots_needed, len(pos_values)):
                    break

    # Fill UTIL slots with best remaining batters
    util_slots = max(0, league.roster_util - keeper_set.positions_filled.get("util", 0))
    if util_slots > 0:
        remaining = [v for v in available if v.player_id not in used_ids and v.player_type == "batter"]
        for v in remaining[:util_slots]:
            total_value += v.value

    return total_value


def solve_keepers_with_pool(
    candidates: list[KeeperDecision],
    constraints: KeeperConstraints,
    league_keeper_ids: set[int],
    valuations: list[Valuation],
    league: LeagueSettings,
) -> KeeperSolution:
    """Find optimal keepers accounting for draft pool depletion."""
    base_pool, _replacement_levels = compute_adjusted_draft_pool(league_keeper_ids, valuations, league)

    valid_sets = _find_valid_sets(candidates, constraints)

    if not valid_sets:
        msg = "no valid keeper sets satisfy the given constraints"
        raise ValueError(msg)

    # Re-score each set: surplus + estimated draft value
    scored = [
        dataclasses.replace(ks, score=ks.total_surplus + _estimated_draft_value(ks, base_pool, league))
        for ks in valid_sets
    ]

    scored.sort(key=lambda s: s.score, reverse=True)

    optimal = scored[0]
    alternatives = scored[1:6]
    sensitivity = _compute_sensitivity(optimal, scored)

    return KeeperSolution(
        optimal=optimal,
        alternatives=alternatives,
        sensitivity=sensitivity,
    )


def parse_league_keepers(
    rows: list[dict[str, str]],
    players: list[Player],
) -> tuple[set[int], list[str]]:
    """Match player names from CSV rows to player IDs."""
    name_to_id: dict[str, int] = {}
    for p in players:
        full_name = f"{p.name_first} {p.name_last}".lower()
        if p.id is not None:
            name_to_id[full_name] = p.id

    matched: set[int] = set()
    unmatched: list[str] = []

    for row in rows:
        name = row["player_name"].strip().lower()
        pid = name_to_id.get(name)
        if pid is not None:
            matched.add(pid)
        else:
            unmatched.append(row["player_name"].strip())

    return matched, unmatched
