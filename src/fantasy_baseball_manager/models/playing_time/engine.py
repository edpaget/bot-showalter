"""OLS fit/predict functions for playing-time projection."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from fantasy_baseball_manager.domain.projection import StatDistribution


@dataclass(frozen=True)
class PlayingTimeCoefficients:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]  # one per feature
    intercept: float
    r_squared: float
    player_type: str  # "batter" or "pitcher"
    alpha: float = 0.0  # ridge penalty; 0 = OLS


@dataclass(frozen=True)
class ResidualPercentiles:
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    count: int
    std: float
    mean_offset: float


@dataclass(frozen=True)
class ResidualBuckets:
    buckets: dict[str, ResidualPercentiles]
    player_type: str  # "batter" or "pitcher"
    fallback_key: str = "all"


def _bucket_key(age: float | int | None, il_days_1: float | None) -> str:
    age_label = "old" if (age is not None and float(age) >= 30) else "young"
    il_label = "injured" if (il_days_1 is not None and float(il_days_1) > 0) else "healthy"
    return f"{age_label}_{il_label}"


def fit_playing_time(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_column: str,
    player_type: str,
    alpha: float = 0.0,
) -> PlayingTimeCoefficients:
    """Fit a ridge regression for playing-time projection.

    When alpha=0.0 this is equivalent to OLS. Rows with None target are
    skipped. None feature values are treated as 0.0.
    """
    filtered = [r for r in rows if r.get(target_column) is not None]

    n = len(filtered)
    k = len(feature_names)

    # Build X matrix with intercept column (column of 1s prepended)
    x_data = np.empty((n, k + 1), dtype=np.float64)
    y_data = np.empty(n, dtype=np.float64)

    for i, row in enumerate(filtered):
        x_data[i, 0] = 1.0  # intercept
        for j, name in enumerate(feature_names):
            val = row.get(name)
            x_data[i, j + 1] = float(val) if val is not None else 0.0
        y_data[i] = float(row[target_column])

    if alpha > 0.0:
        # Ridge closed form: β = (X'X + αI)⁻¹ X'y
        xtx = x_data.T @ x_data
        xty = x_data.T @ y_data
        penalty = alpha * np.eye(k + 1)
        penalty[0, 0] = 0.0  # don't regularize intercept
        beta = np.linalg.solve(xtx + penalty, xty)
    else:
        # OLS via least-squares (handles rank-deficient matrices)
        beta, _, _, _ = np.linalg.lstsq(x_data, y_data, rcond=None)

    intercept = float(beta[0])
    coefficients = tuple(float(b) for b in beta[1:])

    # Compute R²
    y_pred = x_data @ beta
    ss_res = float(np.sum((y_data - y_pred) ** 2))
    ss_tot = float(np.sum((y_data - np.mean(y_data)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return PlayingTimeCoefficients(
        feature_names=tuple(feature_names),
        coefficients=coefficients,
        intercept=intercept,
        r_squared=r_squared,
        player_type=player_type,
        alpha=alpha,
    )


def select_alpha(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_column: str,
    player_type: str,
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_folds: int = 5,
) -> float:
    """Select the best ridge alpha via season-based cross-validation.

    Splits data by season (leave-one-season-out when n_seasons <= n_folds,
    otherwise groups seasons into n_folds folds). Returns the alpha with
    lowest mean RMSE across folds. If only one season exists, returns the
    first alpha.
    """
    # Group rows by season
    season_groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        s = float(row["season"])
        season_groups.setdefault(s, []).append(row)

    seasons = sorted(season_groups.keys())
    if len(seasons) <= 1:
        return alphas[0]

    # Build folds: assign each season to a fold index
    actual_folds = min(n_folds, len(seasons))
    fold_assignments: dict[float, int] = {}
    for i, s in enumerate(seasons):
        fold_assignments[s] = i % actual_folds

    # For each alpha, compute mean RMSE across folds
    best_alpha = alphas[0]
    best_rmse = float("inf")

    for alpha in alphas:
        fold_rmses: list[float] = []
        for fold_idx in range(actual_folds):
            train_rows = [r for r in rows if fold_assignments[float(r["season"])] != fold_idx]
            test_rows = [r for r in rows if fold_assignments[float(r["season"])] == fold_idx]
            if not train_rows or not test_rows:
                continue
            coeff = fit_playing_time(train_rows, feature_names, target_column, player_type, alpha=alpha)
            # Compute RMSE on test fold
            sse = 0.0
            count = 0
            for row in test_rows:
                actual = row.get(target_column)
                if actual is None:
                    continue
                predicted = coeff.intercept
                for name, c in zip(coeff.feature_names, coeff.coefficients):
                    val = row.get(name)
                    predicted += c * (float(val) if val is not None else 0.0)
                sse += (float(actual) - predicted) ** 2
                count += 1
            if count > 0:
                fold_rmses.append((sse / count) ** 0.5)
        if fold_rmses:
            mean_rmse = sum(fold_rmses) / len(fold_rmses)
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_alpha = alpha

    return best_alpha


def compute_residual_buckets(
    rows: list[dict[str, Any]],
    coefficients: PlayingTimeCoefficients,
    target_column: str,
    min_bucket_size: int = 20,
) -> ResidualBuckets:
    """Compute empirical residual percentiles grouped by age/IL buckets."""
    bucket_residuals: dict[str, list[float]] = {"all": []}

    for row in rows:
        actual = row.get(target_column)
        if actual is None:
            continue
        # Raw (unclamped) prediction
        predicted = coefficients.intercept
        for name, coeff in zip(coefficients.feature_names, coefficients.coefficients):
            val = row.get(name)
            predicted += coeff * (float(val) if val is not None else 0.0)
        residual = float(actual) - predicted

        key = _bucket_key(row.get("age"), row.get("il_days_1"))
        bucket_residuals.setdefault(key, []).append(residual)
        bucket_residuals["all"].append(residual)

    buckets: dict[str, ResidualPercentiles] = {}
    for key, residuals in bucket_residuals.items():
        if key != "all" and len(residuals) < min_bucket_size:
            continue
        arr = np.array(residuals, dtype=np.float64)
        p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90]).tolist()
        buckets[key] = ResidualPercentiles(
            p10=p10,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            count=len(residuals),
            std=float(np.std(arr)),
            mean_offset=float(np.mean(arr)),
        )

    return ResidualBuckets(buckets=buckets, player_type=coefficients.player_type)


def predict_playing_time(
    features: dict[str, float | None],
    coefficients: PlayingTimeCoefficients,
    clamp_min: float = 0.0,
    clamp_max: float = 750.0,
) -> float:
    """Predict playing time using fitted coefficients.

    None feature values are treated as 0.0. Result is clamped to [clamp_min, clamp_max].
    """
    total = coefficients.intercept
    for name, coeff in zip(coefficients.feature_names, coefficients.coefficients):
        val = features.get(name)
        total += coeff * (float(val) if val is not None else 0.0)
    return max(clamp_min, min(clamp_max, total))


def predict_playing_time_distribution(
    point_estimate: float,
    features: dict[str, float | None],
    residual_buckets: ResidualBuckets,
    clamp_min: float = 0.0,
    clamp_max: float = 750.0,
) -> StatDistribution:
    """Produce a distributional playing-time estimate from residual buckets."""
    key = _bucket_key(features.get("age"), features.get("il_days_1"))
    percs = residual_buckets.buckets.get(key) or residual_buckets.buckets[residual_buckets.fallback_key]

    def _clamp(val: float) -> float:
        return max(clamp_min, min(clamp_max, val))

    stat = "pa" if residual_buckets.player_type == "batter" else "ip"
    return StatDistribution(
        stat=stat,
        p10=_clamp(point_estimate + percs.p10),
        p25=_clamp(point_estimate + percs.p25),
        p50=_clamp(point_estimate + percs.p50),
        p75=_clamp(point_estimate + percs.p75),
        p90=_clamp(point_estimate + percs.p90),
        mean=_clamp(point_estimate + percs.mean_offset),
        std=percs.std,
        family="residual_bucket",
    )
