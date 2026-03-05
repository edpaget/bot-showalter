"""Single-target quick evaluation and marginal value estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fantasy_baseball_manager.models.sampling import holdout_metrics

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import TrainingBackend


@dataclass(frozen=True)
class QuickEvalResult:
    target: str
    rmse: float
    r_squared: float
    n: int
    baseline_rmse: float | None = None
    delta: float | None = None
    delta_pct: float | None = None


@dataclass(frozen=True)
class TargetDelta:
    target: str
    baseline_rmse: float
    candidate_rmse: float
    delta: float  # candidate_rmse - baseline_rmse (negative = improvement)
    delta_pct: float  # delta / baseline_rmse * 100


@dataclass(frozen=True)
class FeatureSetComparisonResult:
    columns_a: tuple[str, ...]
    columns_b: tuple[str, ...]
    deltas: tuple[TargetDelta, ...]  # baseline_rmse = set A, candidate_rmse = set B
    n_improved: int  # targets where delta < 0 (B better)
    n_total: int
    avg_delta_pct: float
    n_folds: int  # 1 for single holdout, >1 for CV


@dataclass(frozen=True)
class MarginalValueResult:
    candidate: str
    deltas: tuple[TargetDelta, ...]
    n_improved: int  # targets where delta < 0
    n_total: int
    avg_delta_pct: float  # mean of delta_pct across targets


def quick_eval(
    feature_columns: list[str],
    target: str,
    rows_by_season: dict[int, list[dict[str, Any]]],
    train_seasons: list[int],
    holdout_season: int,
    params: dict[str, Any] | None = None,
    *,
    baseline_rmse: float | None = None,
    backend: TrainingBackend,
) -> QuickEvalResult:
    """Train a single target GBM and evaluate on one holdout season.

    Pure computation — no files written, no database changes.
    """
    train_rows = [row for s in train_seasons for row in rows_by_season.get(s, [])]
    holdout_rows = rows_by_season.get(holdout_season, [])

    X_train = backend.extract_features(train_rows, feature_columns)
    y_train = backend.extract_targets(train_rows, [target])

    X_holdout = backend.extract_features(holdout_rows, feature_columns)
    y_holdout = backend.extract_targets(holdout_rows, [target])

    model_params = params or {}
    fitted = backend.fit(X_train, y_train, model_params)

    tv = y_holdout[target]
    X_holdout_filtered = [X_holdout[i] for i in tv.indices]
    y_pred = fitted.predict(target, X_holdout_filtered)

    metrics = holdout_metrics(np.array(tv.values), y_pred)

    delta: float | None = None
    delta_pct: float | None = None
    if baseline_rmse is not None:
        delta = metrics["rmse"] - baseline_rmse
        delta_pct = delta / baseline_rmse * 100

    return QuickEvalResult(
        target=target,
        rmse=metrics["rmse"],
        r_squared=metrics["r_squared"],
        n=int(metrics["n"]),
        baseline_rmse=baseline_rmse,
        delta=delta,
        delta_pct=delta_pct,
    )


def marginal_value(
    candidate_column: str,
    feature_columns: list[str],
    targets: list[str],
    rows_by_season: dict[int, list[dict[str, Any]]],
    train_seasons: list[int],
    holdout_season: int,
    params: dict[str, Any] | None = None,
    *,
    backend: TrainingBackend,
) -> MarginalValueResult:
    """Compare baseline features vs baseline + candidate on identical data.

    Trains two models per target — one with the current feature set, one with
    the candidate column appended — and reports per-target RMSE deltas.

    Pure computation — no files written, no database changes.
    """
    train_rows = [row for s in train_seasons for row in rows_by_season.get(s, [])]
    holdout_rows = rows_by_season.get(holdout_season, [])
    model_params = params or {}

    # Baseline model
    baseline_X_train = backend.extract_features(train_rows, feature_columns)
    baseline_y_train = backend.extract_targets(train_rows, targets)
    baseline_fitted = backend.fit(baseline_X_train, baseline_y_train, model_params)

    baseline_X_holdout = backend.extract_features(holdout_rows, feature_columns)
    baseline_y_holdout = backend.extract_targets(holdout_rows, targets)

    # Candidate model (baseline + candidate column)
    candidate_columns = feature_columns + [candidate_column]
    candidate_X_train = backend.extract_features(train_rows, candidate_columns)
    candidate_y_train = backend.extract_targets(train_rows, targets)
    candidate_fitted = backend.fit(candidate_X_train, candidate_y_train, model_params)

    candidate_X_holdout = backend.extract_features(holdout_rows, candidate_columns)

    # Score each target
    deltas: list[TargetDelta] = []
    for target in targets:
        # Baseline scoring
        b_tv = baseline_y_holdout[target]
        b_X_filtered = [baseline_X_holdout[i] for i in b_tv.indices]
        b_pred = baseline_fitted.predict(target, b_X_filtered)
        b_metrics = holdout_metrics(np.array(b_tv.values), b_pred)

        # Candidate scoring — use same holdout indices for fair comparison
        c_X_filtered = [candidate_X_holdout[i] for i in b_tv.indices]
        c_pred = candidate_fitted.predict(target, c_X_filtered)
        c_metrics = holdout_metrics(np.array(b_tv.values), c_pred)

        delta = c_metrics["rmse"] - b_metrics["rmse"]
        delta_pct = delta / b_metrics["rmse"] * 100 if b_metrics["rmse"] > 0 else 0.0

        deltas.append(
            TargetDelta(
                target=target,
                baseline_rmse=b_metrics["rmse"],
                candidate_rmse=c_metrics["rmse"],
                delta=delta,
                delta_pct=delta_pct,
            )
        )

    n_improved = sum(1 for d in deltas if d.delta < 0)
    avg_delta_pct = float(np.mean([d.delta_pct for d in deltas]))

    return MarginalValueResult(
        candidate=candidate_column,
        deltas=tuple(deltas),
        n_improved=n_improved,
        n_total=len(deltas),
        avg_delta_pct=avg_delta_pct,
    )


def _score_feature_set(
    columns: list[str],
    targets: list[str],
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    model_params: dict[str, Any],
    backend: TrainingBackend,
) -> dict[str, float]:
    """Train on train_rows with given columns, return per-target RMSE on holdout."""
    X_train = backend.extract_features(train_rows, columns)
    y_train = backend.extract_targets(train_rows, targets)
    fitted = backend.fit(X_train, y_train, model_params)

    X_holdout = backend.extract_features(holdout_rows, columns)
    y_holdout = backend.extract_targets(holdout_rows, targets)

    rmses: dict[str, float] = {}
    for target in targets:
        tv = y_holdout[target]
        X_filtered = [X_holdout[i] for i in tv.indices]
        y_pred = fitted.predict(target, X_filtered)
        metrics = holdout_metrics(np.array(tv.values), y_pred)
        rmses[target] = metrics["rmse"]
    return rmses


def compare_feature_sets(
    columns_a: list[str],
    columns_b: list[str],
    targets: list[str],
    rows_by_season: dict[int, list[dict[str, Any]]],
    seasons: list[int],
    params: dict[str, Any] | None = None,
    *,
    backend: TrainingBackend,
) -> FeatureSetComparisonResult:
    """Compare two feature sets on identical data splits.

    With 2 seasons: single-holdout mode (train on first, test on last).
    With 3+ seasons: temporal expanding CV averaged across folds.

    Pure computation — no files written, no database changes.
    """
    sorted_seasons = sorted(seasons)
    if len(sorted_seasons) < 2:
        msg = f"Need at least 2 seasons for comparison (got {len(sorted_seasons)})"
        raise ValueError(msg)

    model_params = params or {}

    # Build folds: list of (train_seasons, holdout_season)
    if len(sorted_seasons) == 2:
        folds = [(sorted_seasons[:1], sorted_seasons[-1])]
    else:
        # Temporal expanding CV: for [s0, s1, ..., sN], yield (train=[s0..si-1], test=si)
        folds = [(sorted_seasons[:i], sorted_seasons[i]) for i in range(1, len(sorted_seasons))]

    # Collect per-target RMSEs across folds
    rmses_a_by_target: dict[str, list[float]] = {t: [] for t in targets}
    rmses_b_by_target: dict[str, list[float]] = {t: [] for t in targets}

    for train_seasons, holdout_season in folds:
        train_rows = [row for s in train_seasons for row in rows_by_season.get(s, [])]
        holdout_rows = rows_by_season.get(holdout_season, [])

        fold_rmses_a = _score_feature_set(columns_a, targets, train_rows, holdout_rows, model_params, backend)
        fold_rmses_b = _score_feature_set(columns_b, targets, train_rows, holdout_rows, model_params, backend)

        for target in targets:
            rmses_a_by_target[target].append(fold_rmses_a[target])
            rmses_b_by_target[target].append(fold_rmses_b[target])

    # Average RMSEs across folds, build TargetDeltas
    deltas: list[TargetDelta] = []
    for target in targets:
        avg_a = float(np.mean(rmses_a_by_target[target]))
        avg_b = float(np.mean(rmses_b_by_target[target]))
        delta = avg_b - avg_a
        delta_pct = delta / avg_a * 100 if avg_a > 0 else 0.0
        deltas.append(
            TargetDelta(
                target=target,
                baseline_rmse=avg_a,
                candidate_rmse=avg_b,
                delta=delta,
                delta_pct=delta_pct,
            )
        )

    n_improved = sum(1 for d in deltas if d.delta < 0)
    avg_delta_pct = float(np.mean([d.delta_pct for d in deltas]))

    return FeatureSetComparisonResult(
        columns_a=tuple(columns_a),
        columns_b=tuple(columns_b),
        deltas=tuple(deltas),
        n_improved=n_improved,
        n_total=len(deltas),
        avg_delta_pct=avg_delta_pct,
        n_folds=len(folds),
    )
