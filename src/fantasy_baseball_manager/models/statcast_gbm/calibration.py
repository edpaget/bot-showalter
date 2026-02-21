from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

_MIN_OBSERVATIONS = 50

# Support both list[list[float]] (from extract_features) and np.ndarray
_FeatureMatrix = list[list[float]] | np.ndarray


def _filter_X(X: _FeatureMatrix, indices: list[int]) -> list[list[float]]:  # noqa: N802
    """Select rows from feature matrix by index."""
    if isinstance(X, np.ndarray):
        return [list(X[i]) for i in indices]
    return [X[i] for i in indices]


def fit_calibrators(
    models: dict[str, Any],
    X_cal: _FeatureMatrix,
    y_cal: dict[str, Any],
) -> dict[str, IsotonicRegression]:
    """Fit per-target isotonic calibrators on holdout (predicted, actual) pairs.

    y_cal values can be either list[float] or TargetVector (with .indices/.values).
    Skips targets with fewer than _MIN_OBSERVATIONS observations.
    """
    calibrators: dict[str, IsotonicRegression] = {}

    for target_name, model in models.items():
        target = y_cal.get(target_name)
        if target is None:
            continue

        # Support both TargetVector (from extract_targets) and plain list[float]
        if hasattr(target, "indices") and hasattr(target, "values"):
            actuals = target.values
            X_filtered = _filter_X(X_cal, target.indices)
        else:
            actuals = target
            X_filtered = X_cal

        if len(actuals) < _MIN_OBSERVATIONS:
            continue

        raw_predictions = model.predict(X_filtered)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_predictions, actuals)
        calibrators[target_name] = iso

    return calibrators


def apply_calibrators(
    raw_predictions: dict[str, list[float]],
    calibrators: dict[str, Any],
) -> dict[str, list[float]]:
    """Apply isotonic calibration to raw predictions.

    Targets without a calibrator are passed through unchanged.
    """
    result: dict[str, list[float]] = {}

    for target_name, preds in raw_predictions.items():
        calibrator = calibrators.get(target_name)
        if calibrator is None:
            result[target_name] = preds
        else:
            calibrated = calibrator.predict(np.array(preds))
            result[target_name] = list(calibrated)

    return result


def save_calibrators(calibrators: dict[str, Any], path: Path) -> None:
    """Save calibrators to disk using joblib."""
    joblib.dump(calibrators, path)


def load_calibrators(path: Path) -> dict[str, Any]:
    """Load calibrators from disk."""
    result: dict[str, Any] = joblib.load(path)
    return result
