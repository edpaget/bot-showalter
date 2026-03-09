"""Distributional ZAR valuation engine.

Runs the full ZAR pipeline at each playing-time scenario level,
then computes probability-weighted expected dollar values per player.
"""

from typing import TYPE_CHECKING

from fantasy_baseball_manager.models.zar.engine import ZarPipelineResult, run_zar_pipeline

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CategoryConfig


def compute_expected_value(scenario_values: list[tuple[float, float]]) -> float:
    """Weighted mean of (dollar_value, weight) pairs."""
    return sum(value * weight for value, weight in scenario_values)


def run_distributional_zar(
    point_stats: list[dict[str, float]],
    scenario_stats: list[list[tuple[dict[str, float], float]]],
    categories: list[CategoryConfig],
    player_positions: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
    budget: float,
    *,
    stdev_overrides: dict[str, float] | None = None,
) -> ZarPipelineResult:
    """Run ZAR at each scenario level and compute expected dollar values.

    Parameters
    ----------
    point_stats:
        Point-estimate stats for each player (used for z-scores/category scores).
    scenario_stats:
        Per-player list of (stats_dict, weight) scenario pairs.
        Players with a single scenario at weight 1.0 degrade to point estimate.
    categories:
        League scoring categories.
    player_positions:
        Position eligibility lists, parallel to point_stats.
    roster_spots:
        Roster slot counts per position.
    num_teams:
        Number of teams in the league.
    budget:
        Total league budget for dollar conversion.
    stdev_overrides:
        Optional per-category stdev overrides passed to the ZAR pipeline.

    Returns
    -------
    ZarPipelineResult with expected dollar values and point-estimate z-scores.
    """
    n_players = len(point_stats)
    if n_players == 0:
        return ZarPipelineResult(z_scores=[], replacement={}, dollar_values=[])

    # Run point-estimate pipeline for z-scores and category scores
    point_result = run_zar_pipeline(
        point_stats,
        categories,
        player_positions,
        roster_spots,
        num_teams,
        budget,
        stdev_overrides=stdev_overrides,
    )

    # Determine number of scenario passes from the max scenario count
    max_scenarios = max(len(s) for s in scenario_stats)

    # Run one ZAR pass per scenario index
    scenario_dollars: list[list[tuple[float, float]]] = [[] for _ in range(n_players)]

    for k in range(max_scenarios):
        # Build pool-wide stats for this scenario index
        scenario_pool: list[dict[str, float]] = []
        weights_at_k: list[float] = []
        for i in range(n_players):
            player_scenarios = scenario_stats[i]
            if k < len(player_scenarios):
                stats_k, weight_k = player_scenarios[k]
            else:
                # Fewer scenarios than max: reuse first (single-scenario player)
                stats_k, weight_k = player_scenarios[0]
                # Weight already accounted for in the single scenario
                weight_k = 0.0  # won't contribute additional weight
            scenario_pool.append(stats_k)
            weights_at_k.append(weight_k)

        # Run full ZAR pipeline on this scenario's pool
        scenario_result = run_zar_pipeline(
            scenario_pool,
            categories,
            player_positions,
            roster_spots,
            num_teams,
            budget,
            stdev_overrides=stdev_overrides,
        )

        # Record each player's dollar value at this scenario
        for i in range(n_players):
            weight = weights_at_k[i]
            if weight > 0.0:
                scenario_dollars[i].append((scenario_result.dollar_values[i], weight))

    # Compute expected dollar values
    expected_dollars: list[float] = []
    for i in range(n_players):
        if scenario_dollars[i]:
            expected_dollars.append(compute_expected_value(scenario_dollars[i]))
        else:
            expected_dollars.append(point_result.dollar_values[i])

    return ZarPipelineResult(
        z_scores=point_result.z_scores,
        replacement=point_result.replacement,
        dollar_values=expected_dollars,
    )
