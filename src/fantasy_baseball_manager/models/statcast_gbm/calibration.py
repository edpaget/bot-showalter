from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

_MIN_OBSERVATIONS = 50

# Support both list[list[float]] (from extract_features) and np.ndarray
_FeatureMatrix = list[list[float]] | np.ndarray


@dataclass(frozen=True)
class AffineCalibrator:
    """Linear calibrator: calibrated = slope * raw + intercept.

    Preserves exact ranking order while correcting bias and scale.
    """

    slope: float
    intercept: float

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self.slope * X + self.intercept


def _filter_X(X: _FeatureMatrix, indices: list[int]) -> list[list[float]]:  # noqa: N802
    """Select rows from feature matrix by index."""
    if isinstance(X, np.ndarray):
        return [list(X[i]) for i in indices]
    return [X[i] for i in indices]


def _extract_actuals_and_features(
    target: Any,
    X_cal: _FeatureMatrix,
) -> tuple[list[float], _FeatureMatrix] | None:
    """Extract actuals and corresponding features from a target.

    Returns None if there are too few observations.
    """
    if hasattr(target, "indices") and hasattr(target, "values"):
        actuals = target.values
        X_filtered = _filter_X(X_cal, target.indices)
    else:
        actuals = target
        X_filtered = X_cal

    if len(actuals) < _MIN_OBSERVATIONS:
        return None
    return actuals, X_filtered


def fit_calibrators(
    models: dict[str, Any],
    X_cal: _FeatureMatrix,
    y_cal: dict[str, Any],
    method: str = "affine",
) -> dict[str, AffineCalibrator | IsotonicRegression]:
    """Fit per-target calibrators on holdout (predicted, actual) pairs.

    method: "affine" (linear, preserves ranking) or "isotonic" (step function).
    y_cal values can be either list[float] or TargetVector (with .indices/.values).
    Skips targets with fewer than _MIN_OBSERVATIONS observations.
    """
    calibrators: dict[str, AffineCalibrator | IsotonicRegression] = {}

    for target_name, model in models.items():
        target = y_cal.get(target_name)
        if target is None:
            continue

        result = _extract_actuals_and_features(target, X_cal)
        if result is None:
            continue
        actuals, X_filtered = result

        raw_predictions = model.predict(X_filtered)

        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_predictions, actuals)
            calibrators[target_name] = iso
        else:
            # Affine: fit y = slope * x + intercept via least squares
            actuals_arr = np.asarray(actuals)
            preds_arr = np.asarray(raw_predictions)
            # np.polyfit(x, y, 1) returns [slope, intercept]
            slope, intercept = np.polyfit(preds_arr, actuals_arr, 1)
            calibrators[target_name] = AffineCalibrator(slope=float(slope), intercept=float(intercept))

    return calibrators


def apply_calibrators(
    raw_predictions: dict[str, list[float]],
    calibrators: dict[str, Any],
) -> dict[str, list[float]]:
    """Apply calibration to raw predictions.

    Works with both AffineCalibrator and IsotonicRegression (both have .predict).
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
