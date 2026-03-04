from collections import defaultdict
from statistics import mean
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PositionScarcity, PositionValueCurve, ScarcityAdjustedPlayer

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings, Valuation


def compute_scarcity(
    valuations: list[Valuation],
    league: LeagueSettings,
) -> list[PositionScarcity]:
    """Compute positional scarcity metrics from valuations and league settings.

    Returns a list of PositionScarcity sorted by slope (most negative first = most scarce).
    """
    if not valuations:
        return []

    all_positions: dict[str, int] = dict(league.positions) | dict(league.pitcher_positions)

    # Group valuations by position, sort each group descending by value
    by_position: dict[str, list[float]] = defaultdict(list)
    for v in valuations:
        if v.position in all_positions:
            by_position[v.position].append(v.value)

    results: list[PositionScarcity] = []
    for position, slots in all_positions.items():
        if slots <= 0 or position not in by_position:
            continue

        values = sorted(by_position[position], reverse=True)
        n = league.teams * slots
        starters = values[:n]

        tier_1_value = mean(starters)

        replacement_value = values[n] if len(values) > n else values[-1]

        total_surplus = sum(v - replacement_value for v in starters)

        # Use starters + some buffer for slope calculation
        buffer_size = min(len(values), n + max(3, n // 4))
        slope_values = values[:buffer_size]
        dropoff_slope = _linear_slope(slope_values)

        steep_rank = _detect_elbow(slope_values)

        results.append(
            PositionScarcity(
                position=position,
                tier_1_value=tier_1_value,
                replacement_value=replacement_value,
                total_surplus=total_surplus,
                dropoff_slope=dropoff_slope,
                steep_rank=steep_rank,
            )
        )

    results.sort(key=lambda ps: ps.dropoff_slope)
    return results


def compute_value_curves(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
) -> list[PositionValueCurve]:
    """Compute per-position value curves showing every player's rank and value.

    Returns one PositionValueCurve per position that has valuations, with the
    cliff point (elbow) detected and marked.
    """
    if not valuations:
        return []

    all_positions: dict[str, int] = dict(league.positions) | dict(league.pitcher_positions)

    # Group valuations by position
    by_position: dict[str, list[Valuation]] = defaultdict(list)
    for v in valuations:
        if v.position in all_positions:
            by_position[v.position].append(v)

    curves: list[PositionValueCurve] = []
    for position, slots in all_positions.items():
        if slots <= 0 or position not in by_position:
            continue

        pos_vals = sorted(by_position[position], key=lambda v: v.value, reverse=True)
        n = league.teams * slots
        top_n = pos_vals[:n]

        values_list: list[tuple[int, str, float]] = []
        for rank, v in enumerate(top_n, start=1):
            name = player_names.get(v.player_id, f"Unknown ({v.player_id})")
            values_list.append((rank, name, v.value))

        # Use the same elbow detection as compute_scarcity
        raw_values = [v.value for v in top_n]
        cliff_rank = _detect_elbow(raw_values)

        curves.append(
            PositionValueCurve(
                position=position,
                values=values_list,
                cliff_rank=cliff_rank,
            )
        )

    return curves


def _linear_slope(values: list[float]) -> float:
    """Compute least-squares slope of values vs rank (0-indexed)."""
    n = len(values)
    if n < 2:
        return 0.0

    x_mean = (n - 1) / 2.0
    y_mean = mean(values)

    numerator = 0.0
    denominator = 0.0
    for i, y in enumerate(values):
        dx = i - x_mean
        numerator += dx * (y - y_mean)
        denominator += dx * dx

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def scarcity_adjusted_value(value: float, scarcity_score: float) -> float:
    """Adjust a player value by positional scarcity: value * (1 + scarcity_score)."""
    return value * (1.0 + scarcity_score)


def _normalize_scarcity_scores(scarcities: list[PositionScarcity]) -> dict[str, float]:
    """Min-max normalize dropoff_slope across positions to [0, 1].

    Most negative slope → 1.0 (most scarce), least negative → 0.0.
    If all slopes are the same, all scores are 0.0.
    """
    if not scarcities:
        return {}

    slopes = [s.dropoff_slope for s in scarcities]
    min_slope = min(slopes)
    max_slope = max(slopes)

    if max_slope == min_slope:
        return {s.position: 0.0 for s in scarcities}

    # Most negative slope gets 1.0, least negative gets 0.0
    return {s.position: (max_slope - s.dropoff_slope) / (max_slope - min_slope) for s in scarcities}


def compute_scarcity_rankings(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
) -> list[ScarcityAdjustedPlayer]:
    """Compute scarcity-adjusted rankings for all players.

    1. Computes positional scarcity via compute_scarcity().
    2. Normalizes scarcity scores to [0, 1].
    3. Adjusts each player's value by their position's scarcity score.
    4. Sorts by adjusted value descending and assigns ranks.
    """
    if not valuations:
        return []

    scarcities = compute_scarcity(valuations, league)
    if not scarcities:
        return []

    scores = _normalize_scarcity_scores(scarcities)

    # Build original rankings (sorted by raw value descending)
    sorted_vals = sorted(valuations, key=lambda v: v.value, reverse=True)
    original_ranks: dict[int, int] = {}
    for rank, v in enumerate(sorted_vals, start=1):
        original_ranks[v.player_id] = rank

    # Build adjusted players
    players: list[ScarcityAdjustedPlayer] = []
    for v in sorted_vals:
        score = scores.get(v.position, 0.0)
        adj_value = scarcity_adjusted_value(v.value, score)
        name = player_names.get(v.player_id, f"Unknown ({v.player_id})")
        players.append(
            ScarcityAdjustedPlayer(
                player_id=v.player_id,
                player_name=name,
                position=v.position,
                player_type=v.player_type,
                original_value=v.value,
                adjusted_value=adj_value,
                original_rank=original_ranks[v.player_id],
                adjusted_rank=0,  # placeholder
                scarcity_score=score,
            )
        )

    # Sort by adjusted value descending and assign adjusted ranks
    players.sort(key=lambda p: p.adjusted_value, reverse=True)
    result: list[ScarcityAdjustedPlayer] = []
    for rank, p in enumerate(players, start=1):
        result.append(
            ScarcityAdjustedPlayer(
                player_id=p.player_id,
                player_name=p.player_name,
                position=p.position,
                player_type=p.player_type,
                original_value=p.original_value,
                adjusted_value=p.adjusted_value,
                original_rank=p.original_rank,
                adjusted_rank=rank,
                scarcity_score=p.scarcity_score,
            )
        )
    return result


def _detect_elbow(values: list[float]) -> int | None:
    """Detect the rank of maximum absolute second derivative (elbow point).

    Returns None if no significant elbow is found.
    """
    if len(values) < 4:
        return None

    # Compute second derivatives (finite differences)
    second_derivatives: list[float] = []
    for i in range(1, len(values) - 1):
        sd = values[i + 1] - 2 * values[i] + values[i - 1]
        second_derivatives.append(sd)

    if not second_derivatives:
        return None

    # Find index and value of max absolute second derivative
    max_idx = 0
    max_abs = abs(second_derivatives[0])
    for i, sd in enumerate(second_derivatives[1:], start=1):
        abs_sd = abs(sd)
        if abs_sd > max_abs:
            max_abs = abs_sd
            max_idx = i

    # No elbow if second derivatives are negligible
    if max_abs < 1e-9:
        return None

    # Threshold: the elbow must be significantly larger than the average
    avg_abs = mean(abs(sd) for sd in second_derivatives)

    if max_abs < avg_abs * 2.0:
        # No significant elbow
        return None

    # max_idx in second_derivatives corresponds to rank max_idx+1 in values (1-indexed)
    return max_idx + 1
