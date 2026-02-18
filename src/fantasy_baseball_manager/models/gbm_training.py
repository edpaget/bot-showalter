import itertools
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from fantasy_baseball_manager.models.sampling import holdout_metrics

logger = logging.getLogger(__name__)


class TargetVector(NamedTuple):
    indices: list[int]
    values: list[float]


def _filter_X(X: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [X[i] for i in indices]


def extract_targets(
    rows: list[dict[str, Any]],
    targets: list[str],
) -> dict[str, TargetVector]:
    result: dict[str, TargetVector] = {t: TargetVector([], []) for t in targets}
    for i, row in enumerate(rows):
        for target in targets:
            if target == "iso":
                slg = row.get("target_slg")
                avg = row.get("target_avg")
                if slg is None or avg is None:
                    continue
                result["iso"].indices.append(i)
                result["iso"].values.append(slg - avg)
            elif target == "hr_per_9":
                hr_val = row.get("target_hr")
                ip_val = row.get("target_ip")
                if hr_val is None or ip_val is None or ip_val == 0:
                    continue
                result["hr_per_9"].indices.append(i)
                result["hr_per_9"].values.append(hr_val * 9 / ip_val)
            elif target == "babip":
                h_val = row.get("target_h")
                hr_val = row.get("target_hr")
                ab_val = row.get("target_ab")
                so_val = row.get("target_so")
                if ab_val is not None:
                    # Batter path: babip = (h - hr) / (ab - so - hr + sf)
                    sf_val = row.get("target_sf")
                    if h_val is None or hr_val is None or so_val is None or sf_val is None:
                        continue
                    denom: float = ab_val - so_val - hr_val + sf_val
                    if denom == 0:
                        continue
                    result["babip"].indices.append(i)
                    result["babip"].values.append((h_val - hr_val) / denom)
                else:
                    # Pitcher path: babip = (h - hr) / (ip * 3 + h - so - hr)
                    ip_val = row.get("target_ip")
                    if h_val is None or hr_val is None or ip_val is None or so_val is None:
                        continue
                    denom = ip_val * 3 + h_val - so_val - hr_val
                    if denom == 0:
                        continue
                    result["babip"].indices.append(i)
                    result["babip"].values.append((h_val - hr_val) / denom)
            else:
                value = row.get(f"target_{target}")
                if value is None:
                    continue
                result[target].indices.append(i)
                result[target].values.append(value)
    logger.debug("Extracted targets: %s", {t: len(tv.values) for t, tv in result.items()})
    return result


def extract_features(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        vector: list[float] = []
        for col in feature_columns:
            value = row.get(col)
            if value is None:
                vector.append(float("nan"))
            else:
                vector.append(float(value))
        matrix.append(vector)
    logger.debug("Feature matrix: %d rows x %d cols", len(matrix), len(feature_columns))
    return matrix


def fit_models(
    X: list[list[float]],
    targets_dict: dict[str, TargetVector],
    model_params: dict[str, Any],
) -> dict[str, HistGradientBoostingRegressor]:
    allowed_params = {"max_iter", "max_depth", "learning_rate", "min_samples_leaf", "max_leaf_nodes"}
    filtered_params = {k: v for k, v in model_params.items() if k in allowed_params}
    logger.info("Fitting %d targets with params %s", len(targets_dict), filtered_params)
    t0 = time.perf_counter()
    models: dict[str, HistGradientBoostingRegressor] = {}
    for target_name, tv in targets_dict.items():
        model = HistGradientBoostingRegressor(**filtered_params)
        model.fit(_filter_X(X, tv.indices), tv.values)
        models[target_name] = model
    logger.info("Fitted %d models in %.1fs", len(models), time.perf_counter() - t0)
    return models


def score_predictions(
    models: dict[str, HistGradientBoostingRegressor],
    X: list[list[float]],
    targets_dict: dict[str, TargetVector],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for target_name, model in models.items():
        tv = targets_dict[target_name]
        y_pred = model.predict(_filter_X(X, tv.indices))
        target_metrics = holdout_metrics(np.array(tv.values), y_pred)
        metrics[f"rmse_{target_name}"] = target_metrics["rmse"]
    return metrics


def compute_permutation_importance(
    models: dict[str, HistGradientBoostingRegressor],
    X: list[list[float]],
    targets_dict: dict[str, TargetVector],
    feature_columns: list[str],
    n_repeats: int = 5,
    rng_seed: int = 42,
) -> dict[str, float]:
    baseline_metrics = score_predictions(models, X, targets_dict)
    baseline_rmses = {t: baseline_metrics[f"rmse_{t}"] for t in targets_dict}
    n_targets = len(targets_dict)
    rng = random.Random(rng_seed)
    n_rows = len(X)

    importances: dict[str, float] = {}
    for j, col_name in enumerate(feature_columns):
        repeat_increases: list[float] = []
        for _ in range(n_repeats):
            X_permuted = [row[:] for row in X]
            perm = list(range(n_rows))
            rng.shuffle(perm)
            for i in range(n_rows):
                X_permuted[i][j] = X[perm[i]][j]
            permuted_metrics = score_predictions(models, X_permuted, targets_dict)
            mean_increase = sum(permuted_metrics[f"rmse_{t}"] - baseline_rmses[t] for t in targets_dict) / n_targets
            repeat_increases.append(mean_increase)
        importances[col_name] = sum(repeat_increases) / n_repeats

    return importances


@dataclass(frozen=True)
class CVFold:
    X_train: list[list[float]]
    y_train: dict[str, TargetVector]
    X_test: list[list[float]]
    y_test: dict[str, TargetVector]


@dataclass(frozen=True)
class GridSearchResult:
    best_params: dict[str, Any]
    best_mean_rmse: float
    per_target_rmse: dict[str, float]
    all_results: list[dict[str, Any]]


def _evaluate_combination(folds: list[CVFold], params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single parameter combination across all CV folds."""
    with threadpool_limits(limits=1, user_api="openmp"):
        target_rmse_sums: dict[str, float] = {}
        n_folds = len(folds)

        for fold in folds:
            models = fit_models(fold.X_train, fold.y_train, params)
            metrics = score_predictions(models, fold.X_test, fold.y_test)
            for key, value in metrics.items():
                target_name = key.removeprefix("rmse_")
                target_rmse_sums[target_name] = target_rmse_sums.get(target_name, 0.0) + value

        per_target_rmse = {t: v / n_folds for t, v in target_rmse_sums.items()}
        mean_rmse = sum(per_target_rmse.values()) / len(per_target_rmse) if per_target_rmse else 0.0

        return {
            "params": params,
            "mean_rmse": mean_rmse,
            "per_target_rmse": per_target_rmse,
        }


def grid_search_cv(
    folds: list[CVFold],
    param_grid: dict[str, list[Any]],
    *,
    max_workers: int | None = None,
) -> GridSearchResult:
    """Exhaustive grid search with pre-computed CV folds.

    For each parameter combination, trains models on each fold's training data,
    scores on each fold's test data, and averages RMSE across folds. Returns
    the combination with the lowest mean RMSE across all targets and folds.

    Args:
        folds: Pre-computed CV folds.
        param_grid: Parameter names mapped to lists of values to try.
        max_workers: Maximum number of parallel worker processes. Defaults to
            the number of CPUs. Set to 1 to disable parallelism.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

    effective_workers = min(max_workers or os.cpu_count() or 1, len(combos))
    logger.info("Grid search: %d combos, %d folds, %d workers", len(combos), len(folds), effective_workers)
    t0 = time.perf_counter()

    if effective_workers <= 1:
        all_results = [_evaluate_combination(folds, params) for params in combos]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(_evaluate_combination, folds, params) for params in combos]
            all_results = [f.result() for f in futures]

    best_mean_rmse = float("inf")
    best_params: dict[str, Any] = {}
    best_per_target: dict[str, float] = {}

    for entry in all_results:
        if entry["mean_rmse"] < best_mean_rmse:
            best_mean_rmse = entry["mean_rmse"]
            best_params = entry["params"]
            best_per_target = entry["per_target_rmse"]

    logger.info(
        "Grid search done in %.1fs: best RMSE=%.4f params=%s", time.perf_counter() - t0, best_mean_rmse, best_params
    )
    return GridSearchResult(
        best_params=best_params,
        best_mean_rmse=best_mean_rmse,
        per_target_rmse=best_per_target,
        all_results=all_results,
    )
