import statistics
from collections import defaultdict

from fantasy_baseball_manager.domain.tier import PlayerTier
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.protocols import PlayerRepo


def generate_tiers(
    valuations: list[Valuation],
    player_repo: PlayerRepo,
    method: str = "gap",
    max_tiers: int = 5,
) -> list[PlayerTier]:
    """Assign tier labels to players grouped by position.

    Args:
        valuations: Per-player valuations with position and value.
        player_repo: Repository for resolving player names.
        method: Clustering method — "gap" or "jenks".
        max_tiers: Maximum number of tiers per position.

    Returns:
        List of PlayerTier assignments sorted by position then rank.
    """
    if not valuations:
        return []

    if method not in ("gap", "jenks"):
        msg = f"Unknown method: {method!r}. Use 'gap' or 'jenks'."
        raise ValueError(msg)

    # Resolve player names in bulk
    player_ids = [v.player_id for v in valuations]
    players_by_id = {p.id: p for p in player_repo.get_by_ids(player_ids) if p.id is not None}

    # Group valuations by position
    by_position: dict[str, list[Valuation]] = defaultdict(list)
    for v in valuations:
        by_position[v.position].append(v)

    result: list[PlayerTier] = []
    for position, pos_vals in sorted(by_position.items()):
        # Sort descending by value
        sorted_vals = sorted(pos_vals, key=lambda v: v.value, reverse=True)

        if method == "gap":
            tier_assignments = _gap_tiers(sorted_vals, max_tiers)
        else:
            tier_assignments = _jenks_tiers(sorted_vals, max_tiers)

        for rank_idx, v in enumerate(sorted_vals):
            player = players_by_id.get(v.player_id)
            name = f"{player.name_first} {player.name_last}" if player else f"Unknown ({v.player_id})"
            result.append(
                PlayerTier(
                    player_id=v.player_id,
                    player_name=name,
                    position=position,
                    tier=tier_assignments[rank_idx],
                    value=v.value,
                    rank=rank_idx + 1,
                )
            )

    return result


def _gap_tiers(sorted_vals: list[Valuation], max_tiers: int) -> list[int]:
    """Gap-based tier assignment: break at gaps > 1.5x median gap."""
    n = len(sorted_vals)
    if n <= 1:
        return [1] * n

    # Compute gaps between consecutive values (descending order)
    gaps = [sorted_vals[i].value - sorted_vals[i + 1].value for i in range(n - 1)]

    median_gap = statistics.median(gaps)
    threshold = 1.5 * median_gap

    # Find break points (0-indexed into gaps list)
    break_indices: list[int] = []
    for i, gap in enumerate(gaps):
        if gap > threshold:
            break_indices.append(i)

    # Cap at max_tiers - 1 break points
    break_indices = break_indices[: max_tiers - 1]

    # Assign tiers
    break_set = set(break_indices)
    tiers: list[int] = []
    current_tier = 1
    for i in range(n):
        tiers.append(current_tier)
        if i in break_set:
            current_tier += 1

    return tiers


def _jenks_tiers(sorted_vals: list[Valuation], max_tiers: int) -> list[int]:
    """Jenks natural breaks (Fisher-Jenks) tier assignment using DP."""
    n = len(sorted_vals)
    if n <= 1:
        return [1] * n

    # Work with ascending values for Jenks
    values = [v.value for v in reversed(sorted_vals)]

    # If all values are equal, single tier
    if values[0] == values[-1]:
        return [1] * n

    k = min(max_tiers, n)

    # Precompute SDCM (sum of squared deviations from class mean)
    # for contiguous subsequences
    # sdcm[i][j] = SDCM of values[i..j]
    sdcm = [[0.0] * n for _ in range(n)]
    for i in range(n):
        running_sum = 0.0
        running_sq = 0.0
        for j in range(i, n):
            running_sum += values[j]
            running_sq += values[j] ** 2
            count = j - i + 1
            mean = running_sum / count
            sdcm[i][j] = running_sq - count * mean**2

    # DP: dp[t][j] = minimum SDCM using t classes for values[0..j]
    # backtrack[t][j] = start index of the t-th class
    dp = [[float("inf")] * n for _ in range(k + 1)]
    backtrack = [[0] * n for _ in range(k + 1)]

    # Base case: 1 class
    for j in range(n):
        dp[1][j] = sdcm[0][j]
        backtrack[1][j] = 0

    # Fill DP
    for t in range(2, k + 1):
        for j in range(t - 1, n):
            for m in range(t - 2, j):
                cost = dp[t - 1][m] + sdcm[m + 1][j]
                if cost < dp[t][j]:
                    dp[t][j] = cost
                    backtrack[t][j] = m + 1

    # Find optimal number of classes (up to k)
    # Use k classes directly (Jenks always uses the requested number)
    num_classes = k

    # Backtrack to find break indices
    breaks: list[int] = []
    j = n - 1
    for t in range(num_classes, 1, -1):
        start = backtrack[t][j]
        breaks.append(start)
        j = start - 1

    breaks.reverse()

    # Assign tiers (values are ascending, but we need descending tier assignment)
    # breaks contains start indices of classes 2..k in ascending value order
    ascending_tiers = [1] * n
    current_class = 1
    for i in range(n):
        if current_class < num_classes and i in breaks:
            current_class += 1
        ascending_tiers[i] = current_class

    # Reverse to match descending value order and flip tier numbering
    # ascending_tiers[0] is lowest value → highest tier number
    descending_tiers: list[int] = []
    for i in range(n - 1, -1, -1):
        descending_tiers.append(num_classes - ascending_tiers[i] + 1)

    return descending_tiers
