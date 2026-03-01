from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    from pathlib import Path

_MIN_OBSERVATIONS = 50

# Support both list[list[float]] (from extract_features) and np.ndarray
_FeatureMatrix = list[list[float]] | np.ndarray

# Per-target (predictions, actuals) pairs from a single CV fold.
FoldData = dict[str, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class AffineCalibrator:
    """Linear calibrator: calibrated = slope * raw + intercept.

    Preserves exact ranking order while correcting bias and scale.
    """

    slope: float
    intercept: float

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return self.slope * X + self.intercept


@dataclass(frozen=True)
class MeanShiftCalibrator:
    """Constant-shift calibrator: calibrated = raw + shift.

    Corrects systematic bias without affecting ranking order or scale.
    """

    shift: float

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return X + self.shift


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


Calibrator = AffineCalibrator | MeanShiftCalibrator | IsotonicRegression


def fit_calibrators(
    models: dict[str, Any],
    X_cal: _FeatureMatrix,
    y_cal: dict[str, Any],
    method: str = "affine",
) -> dict[str, Calibrator]:
    """Fit per-target calibrators on holdout (predicted, actual) pairs.

    method: "affine" (linear), "mean_shift" (constant bias correction),
            or "isotonic" (step function).
    y_cal values can be either list[float] or TargetVector (with .indices/.values).
    Skips targets with fewer than _MIN_OBSERVATIONS observations.
    """
    calibrators: dict[str, Calibrator] = {}

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
        elif method == "mean_shift":
            actuals_arr = np.asarray(actuals)
            preds_arr = np.asarray(raw_predictions)
            shift = float(np.mean(actuals_arr) - np.mean(preds_arr))
            calibrators[target_name] = MeanShiftCalibrator(shift=shift)
        else:
            # Affine: fit y = slope * x + intercept via least squares
            actuals_arr = np.asarray(actuals)
            preds_arr = np.asarray(raw_predictions)
            # np.polyfit(x, y, 1) returns [slope, intercept]
            slope, intercept = np.polyfit(preds_arr, actuals_arr, 1)
            calibrators[target_name] = AffineCalibrator(slope=float(slope), intercept=float(intercept))

    return calibrators


def fit_multifold_calibrators(
    folds: list[FoldData],
    method: str = "affine",
) -> dict[str, Calibrator]:
    """Fit calibrators aggregated across multiple CV folds.

    Affine: average per-fold slopes and intercepts (skipping folds with <50 obs).
    Mean shift: average per-fold shifts (skipping folds with <50 obs).
    Isotonic: pool all fold data and fit a single regression (skip if pooled <50).
    """
    # Collect all target names across folds
    all_targets: set[str] = set()
    for fold in folds:
        all_targets.update(fold.keys())

    calibrators: dict[str, Calibrator] = {}

    for target_name in sorted(all_targets):
        if method == "isotonic":
            all_preds: list[np.ndarray] = []
            all_actuals: list[np.ndarray] = []
            for fold in folds:
                if target_name not in fold:
                    continue
                preds, actuals = fold[target_name]
                all_preds.append(np.asarray(preds))
                all_actuals.append(np.asarray(actuals))
            if not all_preds:
                continue
            pooled_preds = np.concatenate(all_preds)
            pooled_actuals = np.concatenate(all_actuals)
            if len(pooled_preds) < _MIN_OBSERVATIONS:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pooled_preds, pooled_actuals)
            calibrators[target_name] = iso
        elif method == "mean_shift":
            fold_shifts: list[float] = []
            for fold in folds:
                if target_name not in fold:
                    continue
                preds, actuals = fold[target_name]
                preds_arr = np.asarray(preds)
                actuals_arr = np.asarray(actuals)
                if len(preds_arr) < _MIN_OBSERVATIONS:
                    continue
                fold_shifts.append(float(np.mean(actuals_arr) - np.mean(preds_arr)))
            if not fold_shifts:
                continue
            calibrators[target_name] = MeanShiftCalibrator(shift=sum(fold_shifts) / len(fold_shifts))
        else:
            # Affine: fit per-fold, then average
            fold_slopes: list[float] = []
            fold_intercepts: list[float] = []
            for fold in folds:
                if target_name not in fold:
                    continue
                preds, actuals = fold[target_name]
                preds_arr = np.asarray(preds)
                actuals_arr = np.asarray(actuals)
                if len(preds_arr) < _MIN_OBSERVATIONS:
                    continue
                slope, intercept = np.polyfit(preds_arr, actuals_arr, 1)
                fold_slopes.append(float(slope))
                fold_intercepts.append(float(intercept))
            if not fold_slopes:
                continue
            avg_slope = sum(fold_slopes) / len(fold_slopes)
            avg_intercept = sum(fold_intercepts) / len(fold_intercepts)
            calibrators[target_name] = AffineCalibrator(slope=avg_slope, intercept=avg_intercept)

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
