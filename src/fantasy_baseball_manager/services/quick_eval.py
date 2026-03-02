"""Single-target quick evaluation and marginal value estimation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from fantasy_baseball_manager.models.gbm_training import (
    extract_features,
    extract_targets,
    fit_models,
)
from fantasy_baseball_manager.models.sampling import holdout_metrics


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
) -> QuickEvalResult:
    """Train a single target GBM and evaluate on one holdout season.

    Pure computation — no files written, no database changes.
    """
    train_rows = [row for s in train_seasons for row in rows_by_season.get(s, [])]
    holdout_rows = rows_by_season.get(holdout_season, [])

    X_train = extract_features(train_rows, feature_columns)
    y_train = extract_targets(train_rows, [target])

    X_holdout = extract_features(holdout_rows, feature_columns)
    y_holdout = extract_targets(holdout_rows, [target])

    model_params = params or {}
    models = fit_models(X_train, y_train, model_params)

    model = models[target]
    tv = y_holdout[target]
    X_holdout_filtered = [X_holdout[i] for i in tv.indices]
    y_pred = model.predict(X_holdout_filtered)

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
) -> MarginalValueResult:
    """Compare baseline features vs baseline + candidate on identical data.

    Trains two GBMs per target — one with the current feature set, one with
    the candidate column appended — and reports per-target RMSE deltas.

    Pure computation — no files written, no database changes.
    """
    train_rows = [row for s in train_seasons for row in rows_by_season.get(s, [])]
    holdout_rows = rows_by_season.get(holdout_season, [])
    model_params = params or {}

    # Baseline model
    baseline_X_train = extract_features(train_rows, feature_columns)
    baseline_y_train = extract_targets(train_rows, targets)
    baseline_models = fit_models(baseline_X_train, baseline_y_train, model_params)

    baseline_X_holdout = extract_features(holdout_rows, feature_columns)
    baseline_y_holdout = extract_targets(holdout_rows, targets)

    # Candidate model (baseline + candidate column)
    candidate_columns = feature_columns + [candidate_column]
    candidate_X_train = extract_features(train_rows, candidate_columns)
    candidate_y_train = extract_targets(train_rows, targets)
    candidate_models = fit_models(candidate_X_train, candidate_y_train, model_params)

    candidate_X_holdout = extract_features(holdout_rows, candidate_columns)

    # Score each target
    deltas: list[TargetDelta] = []
    for target in targets:
        # Baseline scoring
        b_tv = baseline_y_holdout[target]
        b_X_filtered = [baseline_X_holdout[i] for i in b_tv.indices]
        b_pred = baseline_models[target].predict(b_X_filtered)
        b_metrics = holdout_metrics(np.array(b_tv.values), b_pred)

        # Candidate scoring — use same holdout indices for fair comparison
        c_X_filtered = [candidate_X_holdout[i] for i in b_tv.indices]
        c_pred = candidate_models[target].predict(c_X_filtered)
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
