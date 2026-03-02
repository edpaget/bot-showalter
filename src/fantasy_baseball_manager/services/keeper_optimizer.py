import heapq
import itertools
import math
from collections import Counter

from fantasy_baseball_manager.domain import (
    KeeperConstraints,
    KeeperDecision,
    KeeperSet,
    KeeperSolution,
    SensitivityEntry,
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
