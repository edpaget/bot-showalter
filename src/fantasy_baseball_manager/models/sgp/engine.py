import statistics
from dataclasses import dataclass

from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    StatType,
)
from fantasy_baseball_manager.models.zar.engine import (
    PlayerZScores,
    compute_replacement_level,
    compute_var,
    resolve_numerator,
    run_optimal_pipeline,
    var_to_dollars,
)


@dataclass(frozen=True)
class PlayerSgpScores:
    player_index: int
    category_sgp: dict[str, float]
    composite_sgp: float


@dataclass(frozen=True)
class SgpPipelineResult:
    sgp_scores: list[PlayerSgpScores]
    replacement: dict[str, float]
    dollar_values: list[float]
    assignments: dict[int, str] | None = None


def compute_sgp_scores(
    stats_list: list[dict[str, float]],
    categories: list[CategoryConfig],
    denominators: dict[str, float],
    *,
    use_direct_rates: bool = False,
    volume_weighted: bool = False,
    representative_team: dict[str, tuple[float, float]] | None = None,
) -> list[PlayerSgpScores]:
    """Compute per-category SGP scores for each player.

    Counting stats: sgp = stat_value / denominator
    Rate stats: sgp = (player_rate - baseline) / denom (higher-is-better)
                sgp = (baseline - player_rate) / abs(denom) (lower-is-better)

    No IP/PA multiplication for rate stats — this is the key difference from ZAR.

    When *use_direct_rates* is ``True``, rate categories read the player rate
    directly from ``stats[cat.key]`` instead of deriving from components.
    The baseline becomes the median of direct rates.  Falls back to the
    derived calculation for players missing the key.

    When *volume_weighted* is ``True``, rate-stat baselines use volume-weighted
    means instead of medians, and each player's rate-stat SGP is scaled by
    ``player_volume / avg_volume``.  This prevents relievers with low IP from
    receiving the same rate-stat credit as starters with high IP.
    """
    if not stats_list:
        return []

    # Pre-compute baselines for rate categories
    baselines: dict[str, float] = {}
    for cat in categories:
        if cat.stat_type is StatType.RATE and cat.numerator and cat.denominator:
            if volume_weighted:
                weighted_sum = 0.0
                total_vol = 0.0
                for stats in stats_list:
                    vol = stats.get(cat.denominator, 0.0)
                    if vol > 0.0:
                        if use_direct_rates and cat.key in stats:
                            rate = stats[cat.key]
                        else:
                            rate = resolve_numerator(cat.numerator, stats) / vol
                        weighted_sum += rate * vol
                        total_vol += vol
                baselines[cat.key] = weighted_sum / total_vol if total_vol else 0.0
            else:
                rates: list[float] = []
                for stats in stats_list:
                    denom_val = stats.get(cat.denominator, 0.0)
                    if denom_val > 0.0:
                        if use_direct_rates and cat.key in stats:
                            rates.append(stats[cat.key])
                        else:
                            rates.append(resolve_numerator(cat.numerator, stats) / denom_val)
                baselines[cat.key] = statistics.median(rates) if rates else 0.0

    # Pre-compute average volumes per rate category for volume weighting
    avg_volumes: dict[str, float] = {}
    if volume_weighted:
        for cat in categories:
            if cat.stat_type is StatType.RATE and cat.denominator:
                vols = [s.get(cat.denominator, 0.0) for s in stats_list if s.get(cat.denominator, 0.0) > 0]
                avg_volumes[cat.key] = statistics.mean(vols) if vols else 0.0

    result: list[PlayerSgpScores] = []
    for i, stats in enumerate(stats_list):
        category_sgp: dict[str, float] = {}
        for cat in categories:
            denom = denominators.get(cat.key, 0.0)
            if denom == 0.0:
                category_sgp[cat.key] = 0.0
                continue

            if cat.stat_type is StatType.COUNTING:
                value = resolve_numerator(cat.key, stats)
                category_sgp[cat.key] = value / denom
            elif cat.numerator and cat.denominator:
                denom_val = stats.get(cat.denominator, 0.0)
                if denom_val <= 0.0:
                    category_sgp[cat.key] = 0.0
                    continue

                # Team-impact path: use marginal impact on representative team
                if representative_team is not None and cat.key in representative_team:
                    team_rate, team_vol = representative_team[cat.key]
                    if use_direct_rates and cat.key in stats:
                        player_rate = stats[cat.key]
                    else:
                        player_rate = resolve_numerator(cat.numerator, stats) / denom_val
                    rate_with = (team_rate * team_vol + player_rate * denom_val) / (team_vol + denom_val)
                    marginal = team_rate - rate_with if cat.direction is Direction.LOWER else rate_with - team_rate
                    category_sgp[cat.key] = marginal / abs(denom)
                else:
                    # Existing baseline path (median or volume-weighted mean)
                    if use_direct_rates and cat.key in stats:
                        player_rate = stats[cat.key]
                    else:
                        player_rate = resolve_numerator(cat.numerator, stats) / denom_val
                    baseline = baselines[cat.key]
                    if cat.direction is Direction.LOWER:
                        raw_sgp = (baseline - player_rate) / abs(denom)
                    else:
                        raw_sgp = (player_rate - baseline) / denom
                    if volume_weighted:
                        vol_weight = denom_val / avg_volumes.get(cat.key, 1.0)
                        category_sgp[cat.key] = raw_sgp * vol_weight
                    else:
                        category_sgp[cat.key] = raw_sgp

        composite = sum(category_sgp.values())
        result.append(
            PlayerSgpScores(
                player_index=i,
                category_sgp=category_sgp,
                composite_sgp=composite,
            )
        )
    return result


def run_sgp_pipeline(
    stats_list: list[dict[str, float]],
    categories: list[CategoryConfig],
    denominators: dict[str, float],
    player_positions: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
    budget: float,
    *,
    use_direct_rates: bool = False,
    use_optimal_assignment: bool = True,
    volume_weighted: bool = False,
    representative_team: dict[str, tuple[float, float]] | None = None,
) -> SgpPipelineResult:
    """Run the full SGP pipeline: SGP scores → replacement → VAR → dollars."""
    if not stats_list:
        return SgpPipelineResult(sgp_scores=[], replacement={}, dollar_values=[])

    sgp_scores = compute_sgp_scores(
        stats_list,
        categories,
        denominators,
        use_direct_rates=use_direct_rates,
        volume_weighted=volume_weighted,
        representative_team=representative_team,
    )

    if use_optimal_assignment:
        composite_scores = [s.composite_sgp for s in sgp_scores]
        replacement, dollar_values, assignments = run_optimal_pipeline(
            composite_scores, player_positions, roster_spots, num_teams, budget
        )
        return SgpPipelineResult(
            sgp_scores=sgp_scores, replacement=replacement, dollar_values=dollar_values, assignments=assignments
        )

    # Greedy fallback: reuse ZAR's replacement/VAR/dollars logic
    z_compat = [
        PlayerZScores(
            player_index=s.player_index,
            category_z=s.category_sgp,
            composite_z=s.composite_sgp,
        )
        for s in sgp_scores
    ]

    replacement = compute_replacement_level(z_compat, player_positions, roster_spots, num_teams)
    var_values = compute_var(z_compat, replacement, player_positions)
    roster_spots_total = sum(spots * num_teams for spots in roster_spots.values())
    dollar_values = var_to_dollars(var_values, budget, roster_spots_total=roster_spots_total)

    return SgpPipelineResult(
        sgp_scores=sgp_scores,
        replacement=replacement,
        dollar_values=dollar_values,
    )
