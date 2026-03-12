import statistics
from dataclasses import dataclass

from fantasy_baseball_manager.domain import (
    BudgetSplitMode,
    CategoryConfig,
    Direction,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.models.zar.assignment import assign_positions


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
    *,
    use_direct_rates: bool = False,
) -> list[dict[str, float]]:
    """Convert rate stats to marginal contributions; negate counting stats with LOWER direction.

    When *use_direct_rates* is ``True``, rate categories read the player rate
    directly from ``stats[cat.key]`` (e.g. ``stats["avg"]``) instead of
    deriving it from ``numerator / denominator``.  The baseline becomes a
    volume-weighted mean of direct rates.  Falls back to the derived
    calculation for players missing the key.

    Returns new dicts — input is not mutated.
    """
    if not stats_list:
        return []

    # Pre-compute baselines for rate categories
    baselines: dict[str, float] = {}
    for cat in categories:
        if cat.stat_type is StatType.RATE and cat.numerator and cat.denominator:
            if use_direct_rates:
                # Volume-weighted mean of direct rates
                weighted_sum = 0.0
                total_denom = 0.0
                for s in stats_list:
                    d = s.get(cat.denominator, 0.0)
                    if d > 0.0:
                        rate = s[cat.key] if cat.key in s else resolve_numerator(cat.numerator, s) / d
                        weighted_sum += rate * d
                        total_denom += d
                baselines[cat.key] = weighted_sum / total_denom if total_denom else 0.0
            else:
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
                    if use_direct_rates and cat.key in stats:
                        player_rate = stats[cat.key]
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
    *,
    stdev_overrides: dict[str, float] | None = None,
    category_weights: dict[str, float] | None = None,
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
        if stdev_overrides and key in stdev_overrides:
            stds[key] = stdev_overrides[key]
        else:
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
        if category_weights:
            composite = sum(category_z[k] * category_weights.get(k, 1.0) for k in category_keys)
        else:
            composite = sum(category_z.values())
        result.append(PlayerZScores(player_index=i, category_z=category_z, composite_z=composite))
    return result


def normalize_composite_z(
    z_scores: list[PlayerZScores],
    reference_stdev: float,
) -> list[PlayerZScores]:
    """Scale composite z-scores so their stdev matches *reference_stdev*."""
    if len(z_scores) < 2:
        return z_scores
    pool_stdev = statistics.pstdev([pz.composite_z for pz in z_scores])
    if pool_stdev == 0.0:
        return z_scores
    scale = reference_stdev / pool_stdev
    return [
        PlayerZScores(
            player_index=pz.player_index,
            category_z=pz.category_z,
            composite_z=pz.composite_z * scale,
        )
        for pz in z_scores
    ]


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
            [pz.composite_z for pz, pos in zip(z_scores, positions, strict=True) if position in pos],
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
    for pz, pos_list in zip(z_scores, positions, strict=True):
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
    *,
    roster_spots_total: int | None = None,
) -> list[float]:
    """Convert VAR to auction dollar values.

    When *roster_spots_total* is given, only the top N positive-VAR players
    (where N = min(roster_spots_total, count of positive-VAR players)) share
    the budget.  Non-draftable players receive $0.
    """
    if not var_values:
        return []

    if roster_spots_total is None:
        # Legacy behaviour: every player shares the budget.
        n = len(var_values)
        surplus = total_budget - n * min_bid
        sum_positive = sum(v for v in var_values if v > 0.0)

        if sum_positive <= 0.0 or surplus <= 0.0:
            return [min_bid] * n

        return [min_bid + (v / sum_positive) * surplus if v > 0.0 else min_bid for v in var_values]

    # Draftable-only mode: rank players by VAR descending, pick top N positive.
    indexed = sorted(enumerate(var_values), key=lambda iv: iv[1], reverse=True)
    draftable_indices: set[int] = set()
    for idx, var in indexed:
        if var <= 0.0:
            break
        if len(draftable_indices) >= roster_spots_total:
            break
        draftable_indices.add(idx)

    n_draftable = len(draftable_indices)
    if n_draftable == 0:
        return [0.0] * len(var_values)

    sum_draftable_var = sum(var_values[i] for i in draftable_indices)
    surplus = total_budget - n_draftable * min_bid

    result = [0.0] * len(var_values)
    for i in draftable_indices:
        result[i] = min_bid + (var_values[i] / sum_draftable_var) * surplus
    return result


def assignment_to_dollars(
    var_values: list[float],
    assignments: dict[int, str],
    total_budget: float,
    min_bid: float = 1.0,
) -> list[float]:
    """Convert VAR to dollars using the optimal assignment set.

    All assigned players are draftable (including VAR=0 replacement-level players,
    who receive *min_bid*).  Players with VAR > 0 share the surplus proportionally.
    Unassigned players receive $0.
    """
    if not var_values:
        return []

    n_assigned = len(assignments)
    if n_assigned == 0:
        return [0.0] * len(var_values)

    base_cost = n_assigned * min_bid
    surplus = total_budget - base_cost

    sum_positive_var = sum(var_values[i] for i in assignments if var_values[i] > 0.0)

    result = [0.0] * len(var_values)
    if sum_positive_var > 0.0 and surplus > 0.0:
        for i in assignments:
            if var_values[i] > 0.0:
                result[i] = min_bid + (var_values[i] / sum_positive_var) * surplus
            else:
                result[i] = min_bid
    else:
        # All assigned players at replacement level — distribute evenly
        equal_share = total_budget / n_assigned
        for i in assignments:
            result[i] = equal_share
    return result


def run_optimal_pipeline(
    composite_scores: list[float],
    player_positions: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
    budget: float,
) -> tuple[dict[str, float], list[float], dict[int, str]]:
    """Run optimal assignment pipeline: assign → dollars.

    Returns (replacement_levels, dollar_values, assignments).
    """
    result = assign_positions(composite_scores, player_positions, roster_spots, num_teams)
    dollar_values = assignment_to_dollars(result.var_values, result.assignments, budget)
    return result.replacement, dollar_values, result.assignments


@dataclass(frozen=True)
class ZarPipelineResult:
    z_scores: list[PlayerZScores]
    replacement: dict[str, float]
    dollar_values: list[float]
    assignments: dict[int, str] | None = None


def run_zar_pipeline(
    stats_list: list[dict[str, float]],
    categories: list[CategoryConfig],
    player_positions: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
    budget: float,
    *,
    stdev_overrides: dict[str, float] | None = None,
    category_weights: dict[str, float] | None = None,
    use_direct_rates: bool = False,
    use_optimal_assignment: bool = True,
    reference_composite_stdev: float | None = None,
) -> ZarPipelineResult:
    """Run the full ZAR pipeline: convert → z-score → replacement → VAR → dollars."""
    if not stats_list:
        return ZarPipelineResult(z_scores=[], replacement={}, dollar_values=[])
    category_keys = [c.key for c in categories]
    converted = convert_rate_stats(stats_list, categories, use_direct_rates=use_direct_rates)
    z_scores = compute_z_scores(
        converted, category_keys, stdev_overrides=stdev_overrides, category_weights=category_weights
    )

    if reference_composite_stdev is not None:
        z_scores = normalize_composite_z(z_scores, reference_composite_stdev)

    if use_optimal_assignment:
        composite_scores = [pz.composite_z for pz in z_scores]
        replacement, dollar_values, assignments = run_optimal_pipeline(
            composite_scores, player_positions, roster_spots, num_teams, budget
        )
        return ZarPipelineResult(
            z_scores=z_scores, replacement=replacement, dollar_values=dollar_values, assignments=assignments
        )

    replacement = compute_replacement_level(z_scores, player_positions, roster_spots, num_teams)
    var_values = compute_var(z_scores, replacement, player_positions)
    roster_spots_total = sum(spots * num_teams for spots in roster_spots.values())
    dollar_values = var_to_dollars(var_values, budget, roster_spots_total=roster_spots_total)
    return ZarPipelineResult(z_scores=z_scores, replacement=replacement, dollar_values=dollar_values)


def compute_budget_split(league: LeagueSettings) -> tuple[float, float]:
    """Split total league budget between batting and pitching.

    When ``budget_split`` is ``ROSTER_SPOTS`` (the default), the split is
    proportional to ``roster_batters / (roster_batters + roster_pitchers)``.
    When ``CATEGORIES``, it uses the category count ratio (legacy behaviour).
    """
    total_budget = league.budget * league.teams

    if league.budget_split is BudgetSplitMode.FIXED_RATIO:
        if league.budget_hitter_pct is None:
            raise ValueError("budget_hitter_pct required for FIXED_RATIO mode")
        return (
            total_budget * league.budget_hitter_pct,
            total_budget * (1 - league.budget_hitter_pct),
        )

    if league.budget_split is BudgetSplitMode.ROSTER_SPOTS:
        total_slots = league.roster_batters + league.roster_pitchers
        if total_slots == 0:
            return 0.0, 0.0
        return (
            total_budget * league.roster_batters / total_slots,
            total_budget * league.roster_pitchers / total_slots,
        )

    # CATEGORIES mode — legacy behaviour
    n_bat = len(league.batting_categories)
    n_pitch = len(league.pitching_categories)
    total = n_bat + n_pitch
    if total == 0:
        return 0.0, 0.0
    return total_budget * n_bat / total, total_budget * n_pitch / total
