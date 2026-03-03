"""Validation gate — pre-flight confidence estimator for model changes.

Analyzes per-fold CV results to estimate whether a candidate model
change is likely to pass the full comparison protocol.
"""

import statistics
from dataclasses import dataclass
from typing import Any

from fantasy_baseball_manager.models.gbm_training import (
    extract_features,
    extract_targets,
    fit_models,
    score_predictions,
)
from fantasy_baseball_manager.models.sampling import temporal_expanding_cv


@dataclass(frozen=True)
class PreflightThresholds:
    """Configurable thresholds for pre-flight confidence classification."""

    high_win_rate: float = 0.75
    high_target_pct: float = 0.80
    medium_win_rate: float = 0.60
    medium_target_pct: float = 0.60


@dataclass(frozen=True)
class TargetPreflightDetail:
    """Per-target pre-flight metrics across CV folds."""

    target: str
    win_rate: float
    mean_delta: float
    delta_std: float


@dataclass(frozen=True)
class PreflightResult:
    """Overall pre-flight verdict with per-target breakdown."""

    details: tuple[TargetPreflightDetail, ...]
    confidence: str
    recommendation: str


def preflight_check(
    cv_results: list[dict[str, float]],
    baseline_cv_results: list[dict[str, float]],
    thresholds: PreflightThresholds | None = None,
) -> PreflightResult:
    """Analyze per-fold CV results and estimate confidence that the full gate will pass.

    Args:
        cv_results: Per-fold dicts mapping target name → RMSE for the candidate.
        baseline_cv_results: Per-fold dicts mapping target name → RMSE for the baseline.
        thresholds: Optional custom thresholds for confidence classification.

    Returns:
        PreflightResult with per-target details, confidence level, and recommendation.
    """
    if thresholds is None:
        thresholds = PreflightThresholds()

    n_folds = len(cv_results)
    if n_folds == 0:
        return PreflightResult(details=(), confidence="low", recommendation="skip")

    targets = sorted(cv_results[0].keys())
    details: list[TargetPreflightDetail] = []

    for target in targets:
        wins = sum(1 for i in range(n_folds) if cv_results[i][target] < baseline_cv_results[i][target])
        win_rate = wins / n_folds
        deltas = [cv_results[i][target] - baseline_cv_results[i][target] for i in range(n_folds)]
        mean_delta = statistics.mean(deltas)
        delta_std = statistics.stdev(deltas) if n_folds > 1 else 0.0
        details.append(
            TargetPreflightDetail(
                target=target,
                win_rate=win_rate,
                mean_delta=mean_delta,
                delta_std=delta_std,
            )
        )

    n_targets = len(details)
    if n_targets == 0:
        return PreflightResult(details=(), confidence="low", recommendation="skip")

    high_count = sum(1 for d in details if d.win_rate >= thresholds.high_win_rate)
    medium_count = sum(1 for d in details if d.win_rate >= thresholds.medium_win_rate)

    if high_count / n_targets >= thresholds.high_target_pct:
        confidence = "high"
    elif medium_count / n_targets >= thresholds.medium_target_pct:
        confidence = "medium"
    else:
        confidence = "low"

    recommendation_map = {"high": "proceed", "medium": "marginal", "low": "skip"}
    recommendation = recommendation_map[confidence]

    return PreflightResult(
        details=tuple(details),
        confidence=confidence,
        recommendation=recommendation,
    )


def score_cv_folds(
    columns: list[str],
    targets: list[str],
    rows_by_season: dict[int, list[dict[str, Any]]],
    seasons: list[int],
    params: dict[str, Any],
) -> list[dict[str, float]]:
    """Score a feature set across temporal expanding CV folds.

    Builds temporal expanding CV folds, trains a GBM on each fold's training
    data, and records per-target RMSE on the holdout fold.

    Args:
        columns: Feature column names to use.
        targets: Target names (without ``target_`` prefix).
        rows_by_season: Training data grouped by season.
        seasons: Season years to use for CV splits.
        params: GBM hyperparameters.

    Returns:
        List of per-fold dicts mapping target name → RMSE.
    """
    cv_splits = list(temporal_expanding_cv(seasons))
    fold_results: list[dict[str, float]] = []

    for train_seasons, test_season in cv_splits:
        train_rows = [row for s in train_seasons for row in rows_by_season.get(s, [])]
        test_rows = rows_by_season.get(test_season, [])

        X_train = extract_features(train_rows, columns)
        y_train = extract_targets(train_rows, targets)
        X_test = extract_features(test_rows, columns)
        y_test = extract_targets(test_rows, targets)

        models = fit_models(X_train, y_train, params)
        metrics = score_predictions(models, X_test, y_test)

        fold_dict: dict[str, float] = {}
        for key, value in metrics.items():
            target_name = key.removeprefix("rmse_")
            fold_dict[target_name] = value
        fold_results.append(fold_dict)

    return fold_results
