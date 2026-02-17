import statistics
from dataclasses import dataclass

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    StatType,
)


@dataclass(frozen=True)
class PlayerZScores:
    player_index: int
    category_z: dict[str, float]
    composite_z: float


@dataclass(frozen=True)
class PlayerValue:
    player_index: int
    var: float
    dollars: float


def resolve_numerator(expression: str, stats: dict[str, float]) -> float:
    """Parse compound numerator expressions like ``"bb+h"`` and sum the values."""
    parts = expression.split("+")
    return sum(stats.get(part.strip(), 0.0) for part in parts)


def convert_rate_stats(
    stats_list: list[dict[str, float]],
    categories: list[CategoryConfig],
) -> list[dict[str, float]]:
    """Convert rate stats to marginal contributions; negate counting stats with LOWER direction.

    Returns new dicts — input is not mutated.
    """
    if not stats_list:
        return []

    # Pre-compute baselines for rate categories (ratio of sums)
    baselines: dict[str, float] = {}
    for cat in categories:
        if cat.stat_type is StatType.RATE and cat.numerator and cat.denominator:
            total_num = sum(resolve_numerator(cat.numerator, s) for s in stats_list)
            total_denom = sum(s.get(cat.denominator, 0.0) for s in stats_list)
            baselines[cat.key] = total_num / total_denom if total_denom else 0.0

    result: list[dict[str, float]] = []
    for stats in stats_list:
        row: dict[str, float] = {}
        for cat in categories:
            if cat.stat_type is StatType.COUNTING:
                value = resolve_numerator(cat.key, stats)
                row[cat.key] = -value if cat.direction is Direction.LOWER else value
            elif cat.numerator and cat.denominator:
                denom = stats.get(cat.denominator, 0.0)
                if denom == 0.0:
                    row[cat.key] = 0.0
                else:
                    player_rate = resolve_numerator(cat.numerator, stats) / denom
                    baseline = baselines[cat.key]
                    if cat.direction is Direction.LOWER:
                        row[cat.key] = (baseline - player_rate) * denom
                    else:
                        row[cat.key] = (player_rate - baseline) * denom
        result.append(row)
    return result


def compute_z_scores(
    stats_list: list[dict[str, float]],
    category_keys: list[str],
) -> list[PlayerZScores]:
    """Compute per-category z-scores and composite z for each player."""
    if not stats_list:
        return []

    n = len(stats_list)

    # Per-category: compute mean and (population) std dev
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in category_keys:
        values = [s.get(key, 0.0) for s in stats_list]
        means[key] = statistics.mean(values)
        stds[key] = statistics.pstdev(values) if n > 1 else 0.0

    result: list[PlayerZScores] = []
    for i, stats in enumerate(stats_list):
        category_z: dict[str, float] = {}
        for key in category_keys:
            std = stds[key]
            if std == 0.0:
                category_z[key] = 0.0
            else:
                category_z[key] = (stats.get(key, 0.0) - means[key]) / std
        composite = sum(category_z.values())
        result.append(PlayerZScores(player_index=i, category_z=category_z, composite_z=composite))
    return result


def compute_replacement_level(
    z_scores: list[PlayerZScores],
    positions: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
) -> dict[str, float]:
    """Compute replacement-level composite z per position."""
    if not z_scores:
        return {}

    result: dict[str, float] = {}
    for position, spots in roster_spots.items():
        eligible = sorted(
            [pz.composite_z for pz, pos in zip(z_scores, positions) if position in pos],
            reverse=True,
        )
        if not eligible:
            result[position] = 0.0
            continue
        draftable = spots * num_teams
        if draftable >= len(eligible):
            result[position] = eligible[-1]
        else:
            result[position] = eligible[draftable]
    return result


def compute_var(
    z_scores: list[PlayerZScores],
    replacement: dict[str, float],
    positions: list[list[str]],
) -> list[float]:
    """Compute value above replacement for each player using their best position."""
    if not z_scores:
        return []

    max_replacement = max(replacement.values()) if replacement else 0.0

    result: list[float] = []
    for pz, pos_list in zip(z_scores, positions):
        if pos_list:
            # Best position = lowest replacement level → highest VAR
            best_repl = min(
                (replacement[p] for p in pos_list if p in replacement),
                default=max_replacement,
            )
        else:
            best_repl = max_replacement
        result.append(pz.composite_z - best_repl)
    return result


def var_to_dollars(
    var_values: list[float],
    total_budget: float,
    min_bid: float = 1.0,
) -> list[float]:
    """Convert VAR to auction dollar values."""
    if not var_values:
        return []

    n = len(var_values)
    surplus = total_budget - n * min_bid
    sum_positive = sum(v for v in var_values if v > 0.0)

    if sum_positive <= 0.0 or surplus <= 0.0:
        return [min_bid] * n

    return [min_bid + (v / sum_positive) * surplus if v > 0.0 else min_bid for v in var_values]
