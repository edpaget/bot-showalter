"""Single-target quick evaluation for rapid feature hypothesis testing."""

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
