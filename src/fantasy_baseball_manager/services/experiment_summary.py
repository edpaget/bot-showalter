from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    ExplorationSummary,
    FeatureExplorationResult,
    TargetExplorationResult,
)
from fantasy_baseball_manager.repos import ExperimentFilter

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Experiment
    from fantasy_baseball_manager.repos import ExperimentRepo


def _avg_delta_pct(exp: Experiment) -> float:
    """Compute mean delta_pct across all targets."""
    if not exp.target_results:
        return 0.0
    return mean(tr.delta_pct for tr in exp.target_results.values())


def summarize_exploration(
    repo: ExperimentRepo,
    model: str,
    player_type: str,
) -> ExplorationSummary:
    experiments = repo.list(ExperimentFilter(model=model, player_type=player_type))

    if not experiments:
        return ExplorationSummary(
            model=model,
            player_type=player_type,
            total_experiments=0,
            features_tested=[],
            targets_explored=[],
            best_experiment_id=None,
            best_experiment_delta_pct=None,
        )

    # Aggregate features
    feature_stats: dict[str, list[tuple[float, int]]] = {}
    for exp in experiments:
        avg = _avg_delta_pct(exp)
        all_features = exp.feature_diff.get("added", []) + exp.feature_diff.get("removed", [])
        for feat in all_features:
            feature_stats.setdefault(feat, []).append((avg, exp.id))  # type: ignore[arg-type]

    features_tested = []
    for feat, entries in sorted(feature_stats.items()):
        best_entry = min(entries, key=lambda e: e[0])
        features_tested.append(
            FeatureExplorationResult(
                feature=feat,
                best_delta_pct=best_entry[0],
                best_experiment_id=best_entry[1],
                times_tested=len(entries),
            )
        )

    # Aggregate targets
    target_stats: dict[str, list[tuple[float, float, float, int]]] = {}
    for exp in experiments:
        for target, tr in exp.target_results.items():
            target_stats.setdefault(target, []).append(
                (tr.rmse, tr.delta_pct, tr.delta_pct, exp.id)  # type: ignore[arg-type]
            )

    targets_explored = []
    for target, entries in sorted(target_stats.items()):
        best_by_delta = min(entries, key=lambda e: e[1])
        best_by_rmse = min(entries, key=lambda e: e[0])
        targets_explored.append(
            TargetExplorationResult(
                target=target,
                best_rmse=best_by_rmse[0],
                best_delta_pct=best_by_delta[1],
                best_experiment_id=best_by_delta[3],
                experiments_count=len(entries),
            )
        )

    # Overall best experiment (lowest avg delta_pct)
    best_exp = min(experiments, key=_avg_delta_pct)

    return ExplorationSummary(
        model=model,
        player_type=player_type,
        total_experiments=len(experiments),
        features_tested=features_tested,
        targets_explored=targets_explored,
        best_experiment_id=best_exp.id,
        best_experiment_delta_pct=_avg_delta_pct(best_exp),
    )
