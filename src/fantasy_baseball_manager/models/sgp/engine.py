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
    """
    if not stats_list:
        return []

    # Pre-compute median rates for rate categories
    baselines: dict[str, float] = {}
    for cat in categories:
        if cat.stat_type is StatType.RATE and cat.numerator and cat.denominator:
            rates: list[float] = []
            for stats in stats_list:
                denom_val = stats.get(cat.denominator, 0.0)
                if denom_val > 0.0:
                    if use_direct_rates and cat.key in stats:
                        rates.append(stats[cat.key])
                    else:
                        rates.append(resolve_numerator(cat.numerator, stats) / denom_val)
            baselines[cat.key] = statistics.median(rates) if rates else 0.0

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
                if use_direct_rates and cat.key in stats:
                    player_rate = stats[cat.key]
                else:
                    player_rate = resolve_numerator(cat.numerator, stats) / denom_val
                baseline = baselines[cat.key]
                if cat.direction is Direction.LOWER:
                    category_sgp[cat.key] = (baseline - player_rate) / abs(denom)
                else:
                    category_sgp[cat.key] = (player_rate - baseline) / denom

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
) -> SgpPipelineResult:
    """Run the full SGP pipeline: SGP scores → replacement → VAR → dollars."""
    if not stats_list:
        return SgpPipelineResult(sgp_scores=[], replacement={}, dollar_values=[])

    sgp_scores = compute_sgp_scores(stats_list, categories, denominators, use_direct_rates=use_direct_rates)

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
