"""OLS fit/predict functions for playing-time projection."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PlayingTimeCoefficients:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]  # one per feature
    intercept: float
    r_squared: float
    player_type: str  # "batter" or "pitcher"


def fit_playing_time(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target_column: str,
    player_type: str,
) -> PlayingTimeCoefficients:
    """Fit an OLS regression for playing-time projection.

    Rows with None target are skipped. None feature values are treated as 0.0.
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

    # Solve y = X @ beta via least squares
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
    )


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
