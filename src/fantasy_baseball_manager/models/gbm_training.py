import itertools
import random
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from fantasy_baseball_manager.models.sampling import holdout_metrics


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
    return matrix


def fit_models(
    X: list[list[float]],
    targets_dict: dict[str, TargetVector],
    model_params: dict[str, Any],
) -> dict[str, HistGradientBoostingRegressor]:
    allowed_params = {"max_iter", "max_depth", "learning_rate", "min_samples_leaf", "max_leaf_nodes"}
    filtered_params = {k: v for k, v in model_params.items() if k in allowed_params}
    models: dict[str, HistGradientBoostingRegressor] = {}
    for target_name, tv in targets_dict.items():
        model = HistGradientBoostingRegressor(**filtered_params)
        model.fit(_filter_X(X, tv.indices), tv.values)
        models[target_name] = model
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


def grid_search_cv(
    folds: list[CVFold],
    param_grid: dict[str, list[Any]],
) -> GridSearchResult:
    """Exhaustive grid search with pre-computed CV folds.

    For each parameter combination, trains models on each fold's training data,
    scores on each fold's test data, and averages RMSE across folds. Returns
    the combination with the lowest mean RMSE across all targets and folds.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    all_results: list[dict[str, Any]] = []
    best_mean_rmse = float("inf")
    best_params: dict[str, Any] = {}
    best_per_target: dict[str, float] = {}

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))

        # Collect per-target RMSE sums across folds
        target_rmse_sums: dict[str, float] = {}
        n_folds = len(folds)

        for fold in folds:
            models = fit_models(fold.X_train, fold.y_train, params)
            metrics = score_predictions(models, fold.X_test, fold.y_test)
            for key, value in metrics.items():
                # key is "rmse_<target>"
                target_name = key.removeprefix("rmse_")
                target_rmse_sums[target_name] = target_rmse_sums.get(target_name, 0.0) + value

        per_target_rmse = {t: v / n_folds for t, v in target_rmse_sums.items()}
        mean_rmse = sum(per_target_rmse.values()) / len(per_target_rmse) if per_target_rmse else 0.0

        entry: dict[str, Any] = {
            "params": params,
            "mean_rmse": mean_rmse,
            "per_target_rmse": per_target_rmse,
        }
        all_results.append(entry)

        if mean_rmse < best_mean_rmse:
            best_mean_rmse = mean_rmse
            best_params = params
            best_per_target = per_target_rmse

    return GridSearchResult(
        best_params=best_params,
        best_mean_rmse=best_mean_rmse,
        per_target_rmse=best_per_target,
        all_results=all_results,
    )
