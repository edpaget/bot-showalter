import itertools
import logging
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from fantasy_baseball_manager.models.protocols import TargetComparison, ValidationResult
from fantasy_baseball_manager.models.sampling import holdout_metrics

logger = logging.getLogger(__name__)


class TargetVector(NamedTuple):
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class FeatureImportance:
    mean: float
    se: float


@dataclass(frozen=True)
class CorrelationGroup:
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class GroupedImportanceResult:
    groups: tuple[CorrelationGroup, ...]
    group_importance: dict[str, FeatureImportance]
    feature_importance: dict[str, FeatureImportance]


def _find_correlated_groups(
    X: list[list[float]],
    feature_columns: list[str],
    threshold: float = 0.70,
) -> list[CorrelationGroup]:
    n_features = len(feature_columns)
    if n_features == 0:
        return []

    arr = np.array(X)
    # Impute NaN with column means for correlation only
    col_means = np.nanmean(arr, axis=0)
    for j in range(n_features):
        mask = np.isnan(arr[:, j])
        if mask.any():
            arr[mask, j] = col_means[j]

    corr = np.corrcoef(arr, rowvar=False)
    if n_features == 1:
        corr = corr.reshape(1, 1)

    # Union-Find
    parent = list(range(n_features))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n_features):
        for j in range(i + 1, n_features):
            c = corr[i][j]
            if not np.isnan(c) and abs(c) > threshold:
                union(i, j)

    # Collect components
    components: dict[int, list[int]] = {}
    for i in range(n_features):
        root = find(i)
        components.setdefault(root, []).append(i)

    # Sort components by first member index
    sorted_components = sorted(components.values(), key=lambda idxs: idxs[0])

    groups: list[CorrelationGroup] = []
    group_counter = 0
    for idxs in sorted_components:
        members = tuple(feature_columns[i] for i in idxs)
        if len(members) == 1:
            name = members[0]
        else:
            name = f"group_{group_counter}"
            group_counter += 1
        groups.append(CorrelationGroup(name=name, members=members))

    return groups


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
    n_repeats: int = 20,
    rng_seed: int = 42,
) -> dict[str, FeatureImportance]:
    baseline_metrics = score_predictions(models, X, targets_dict)
    baseline_rmses = {t: baseline_metrics[f"rmse_{t}"] for t in targets_dict}
    n_targets = len(targets_dict)
    rng = random.Random(rng_seed)
    n_rows = len(X)

    importances: dict[str, FeatureImportance] = {}
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
        mean_val = sum(repeat_increases) / n_repeats
        se_val = statistics.stdev(repeat_increases) / math.sqrt(n_repeats) if n_repeats > 1 else 0.0
        importances[col_name] = FeatureImportance(mean=mean_val, se=se_val)

    return importances


def compute_grouped_permutation_importance(
    models: dict[str, HistGradientBoostingRegressor],
    X: list[list[float]],
    targets_dict: dict[str, TargetVector],
    feature_columns: list[str],
    n_repeats: int = 20,
    rng_seed: int = 42,
    correlation_threshold: float = 0.70,
) -> GroupedImportanceResult:
    groups = _find_correlated_groups(X, feature_columns, correlation_threshold)

    # Individual importance
    individual = compute_permutation_importance(
        models, X, targets_dict, feature_columns, n_repeats=n_repeats, rng_seed=rng_seed
    )

    # Group importance
    baseline_metrics = score_predictions(models, X, targets_dict)
    baseline_rmses = {t: baseline_metrics[f"rmse_{t}"] for t in targets_dict}
    n_targets = len(targets_dict)
    n_rows = len(X)
    rng = random.Random(rng_seed)

    col_index = {col: idx for idx, col in enumerate(feature_columns)}
    group_importance: dict[str, FeatureImportance] = {}

    for group in groups:
        if len(group.members) == 1:
            # Singleton: copy individual importance
            group_importance[group.name] = individual[group.members[0]]
        else:
            # Multi-member: shuffle all member columns simultaneously
            member_indices = [col_index[m] for m in group.members]
            repeat_increases: list[float] = []
            for _ in range(n_repeats):
                X_permuted = [row[:] for row in X]
                perm = list(range(n_rows))
                rng.shuffle(perm)
                for i in range(n_rows):
                    for j in member_indices:
                        X_permuted[i][j] = X[perm[i]][j]
                permuted_metrics = score_predictions(models, X_permuted, targets_dict)
                mean_increase = sum(permuted_metrics[f"rmse_{t}"] - baseline_rmses[t] for t in targets_dict) / n_targets
                repeat_increases.append(mean_increase)
            mean_val = sum(repeat_increases) / n_repeats
            se_val = statistics.stdev(repeat_increases) / math.sqrt(n_repeats) if n_repeats > 1 else 0.0
            group_importance[group.name] = FeatureImportance(mean=mean_val, se=se_val)

    return GroupedImportanceResult(
        groups=tuple(groups),
        group_importance=group_importance,
        feature_importance=individual,
    )


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


def identify_prune_candidates(
    result: GroupedImportanceResult,
) -> list[str]:
    """Identify features whose group CI upper bound (mean + 2*SE) <= 0."""
    candidates: list[str] = []
    for group in result.groups:
        gi = result.group_importance[group.name]
        ci_upper = gi.mean + 2 * gi.se
        if ci_upper <= 0:
            candidates.extend(group.members)
    return sorted(candidates)


def validate_pruning(
    full_models: dict[str, HistGradientBoostingRegressor],
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    feature_columns: list[str],
    prune_set: list[str],
    targets: list[str],
    model_params: dict[str, Any],
    player_type: str,
    max_degradation_pct: float = 5.0,
) -> ValidationResult:
    """Train a pruned model and compare holdout RMSE per-target against the full model."""
    prune_set_s = set(prune_set)
    pruned_cols = [c for c in feature_columns if c not in prune_set_s]

    if not pruned_cols:
        logger.warning("All features prunable — skipping validation for %s", player_type)
        return ValidationResult(
            player_type=player_type,
            comparisons=(),
            pruned_features=tuple(sorted(prune_set)),
            n_improved=0,
            n_degraded=0,
            max_degradation_pct=0.0,
            go=True,
        )

    if not prune_set:
        return ValidationResult(
            player_type=player_type,
            comparisons=(),
            pruned_features=(),
            n_improved=0,
            n_degraded=0,
            max_degradation_pct=0.0,
            go=True,
        )

    # Score full models on holdout
    full_X_holdout = extract_features(holdout_rows, feature_columns)
    full_y_holdout = extract_targets(holdout_rows, targets)
    full_metrics = score_predictions(full_models, full_X_holdout, full_y_holdout)

    # Train pruned models
    pruned_X_train = extract_features(train_rows, pruned_cols)
    pruned_y_train = extract_targets(train_rows, targets)
    pruned_models = fit_models(pruned_X_train, pruned_y_train, model_params)

    # Score pruned models on holdout
    pruned_X_holdout = extract_features(holdout_rows, pruned_cols)
    pruned_y_holdout = extract_targets(holdout_rows, targets)
    pruned_metrics = score_predictions(pruned_models, pruned_X_holdout, pruned_y_holdout)

    # Build comparisons
    comparisons: list[TargetComparison] = []
    n_improved = 0
    n_degraded = 0
    max_deg = 0.0

    for target in targets:
        full_rmse = full_metrics[f"rmse_{target}"]
        pruned_rmse = pruned_metrics[f"rmse_{target}"]
        delta_pct = (pruned_rmse - full_rmse) / full_rmse * 100 if full_rmse > 0 else 0.0
        comparisons.append(
            TargetComparison(
                target=target,
                full_rmse=full_rmse,
                pruned_rmse=pruned_rmse,
                delta_pct=delta_pct,
            )
        )
        if pruned_rmse < full_rmse:
            n_improved += 1
        elif pruned_rmse > full_rmse:
            n_degraded += 1
            if delta_pct > max_deg:
                max_deg = delta_pct

    go = (n_improved > n_degraded) and (max_deg <= max_degradation_pct)

    return ValidationResult(
        player_type=player_type,
        comparisons=tuple(comparisons),
        pruned_features=tuple(sorted(prune_set)),
        n_improved=n_improved,
        n_degraded=n_degraded,
        max_degradation_pct=max_deg,
        go=go,
    )
