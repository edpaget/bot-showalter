import math
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor


def extract_targets(
    rows: list[dict[str, Any]],
    targets: list[str],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {t: [] for t in targets}
    for row in rows:
        for target in targets:
            if target == "iso":
                slg = row.get("target_slg")
                avg = row.get("target_avg")
                if slg is None or avg is None:
                    continue
                result["iso"].append(slg - avg)
            elif target == "babip":
                h_val = row.get("target_h")
                hr_val = row.get("target_hr")
                ab_val = row.get("target_ab")
                so_val = row.get("target_so")
                sf_val = row.get("target_sf")
                if h_val is None or hr_val is None or ab_val is None or so_val is None or sf_val is None:
                    continue
                denom: float = ab_val - so_val - hr_val + sf_val
                if denom == 0:
                    continue
                result["babip"].append((h_val - hr_val) / denom)
            else:
                value = row.get(f"target_{target}")
                if value is None:
                    continue
                result[target].append(value)
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
    targets_dict: dict[str, list[float]],
    model_params: dict[str, Any],
) -> dict[str, HistGradientBoostingRegressor]:
    allowed_params = {"max_iter", "max_depth", "learning_rate", "min_samples_leaf", "max_leaf_nodes"}
    filtered_params = {k: v for k, v in model_params.items() if k in allowed_params}
    models: dict[str, HistGradientBoostingRegressor] = {}
    for target_name, y in targets_dict.items():
        model = HistGradientBoostingRegressor(**filtered_params)
        model.fit(X, y)
        models[target_name] = model
    return models


def score_predictions(
    models: dict[str, HistGradientBoostingRegressor],
    X: list[list[float]],
    targets_dict: dict[str, list[float]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for target_name, model in models.items():
        y_true = targets_dict[target_name]
        y_pred = model.predict(X)
        mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred, strict=True)) / len(y_true)
        metrics[f"rmse_{target_name}"] = math.sqrt(mse)
    return metrics
