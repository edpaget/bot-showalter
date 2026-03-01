from typing import TYPE_CHECKING

import numpy as np
import pytest

from fantasy_baseball_manager.models.statcast_gbm.calibration import (
    AffineCalibrator,
    FoldData,
    apply_calibrators,
    fit_calibrators,
    fit_multifold_calibrators,
    load_calibrators,
    save_calibrators,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestFitCalibratorsIsotonic:
    def test_fits_calibrators_per_target(self) -> None:
        models = {"avg": _FakeModel(bias=-0.010), "obp": _FakeModel(bias=-0.015)}
        n = 100
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        y_cal = {
            "avg": [0.270 + i * 0.001 for i in range(n)],
            "obp": [0.340 + i * 0.001 for i in range(n)],
        }

        calibrators = fit_calibrators(models, X_cal, y_cal, method="isotonic")

        assert "avg" in calibrators
        assert "obp" in calibrators

    def test_skips_target_with_too_few_observations(self) -> None:
        models = {"avg": _FakeModel(bias=0.0)}
        X_cal = np.zeros((10, 5))  # < 50 observations
        y_cal = {"avg": [0.270] * 10}

        calibrators = fit_calibrators(models, X_cal, y_cal, method="isotonic")

        assert "avg" not in calibrators

    def test_calibrator_reduces_systematic_bias(self) -> None:
        model = _FakeModel(bias=-0.020)
        models = {"avg": model}
        n = 200
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        actuals = [0.250 + i * 0.001 for i in range(n)]
        y_cal = {"avg": actuals}

        calibrators = fit_calibrators(models, X_cal, y_cal, method="isotonic")

        raw_preds = {"avg": list(model.predict(X_cal))}
        calibrated = apply_calibrators(raw_preds, calibrators)

        raw_errors = [abs(raw_preds["avg"][i] - actuals[i]) for i in range(n)]
        cal_errors = [abs(calibrated["avg"][i] - actuals[i]) for i in range(n)]
        assert sum(cal_errors) < sum(raw_errors)


class TestFitCalibratorsAffine:
    def test_default_method_is_affine(self) -> None:
        models = {"avg": _FakeModel(bias=-0.010)}
        n = 100
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        y_cal = {"avg": [0.270 + i * 0.001 for i in range(n)]}

        calibrators = fit_calibrators(models, X_cal, y_cal)

        assert isinstance(calibrators["avg"], AffineCalibrator)

    def test_affine_preserves_ranking(self) -> None:
        model = _FakeModel(bias=-0.020)
        models = {"avg": model}
        n = 200
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        y_cal = {"avg": [0.250 + i * 0.001 for i in range(n)]}

        calibrators = fit_calibrators(models, X_cal, y_cal)

        # Apply to new data — all outputs should be unique and in the same order
        raw_preds = list(np.linspace(0.230, 0.450, 50))
        result = apply_calibrators({"avg": raw_preds}, calibrators)
        calibrated = result["avg"]

        # Verify strict monotonicity (no ties)
        for i in range(len(calibrated) - 1):
            assert calibrated[i] < calibrated[i + 1]

    def test_affine_reduces_bias(self) -> None:
        model = _FakeModel(bias=-0.020)
        models = {"avg": model}
        n = 200
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        actuals = [0.250 + i * 0.001 for i in range(n)]
        y_cal = {"avg": actuals}

        calibrators = fit_calibrators(models, X_cal, y_cal)

        raw_preds = {"avg": list(model.predict(X_cal))}
        calibrated = apply_calibrators(raw_preds, calibrators)

        raw_errors = [abs(raw_preds["avg"][i] - actuals[i]) for i in range(n)]
        cal_errors = [abs(calibrated["avg"][i] - actuals[i]) for i in range(n)]
        assert sum(cal_errors) < sum(raw_errors)

    def test_affine_skips_too_few_observations(self) -> None:
        models = {"avg": _FakeModel(bias=0.0)}
        X_cal = np.zeros((10, 5))
        y_cal = {"avg": [0.270] * 10}

        calibrators = fit_calibrators(models, X_cal, y_cal)

        assert "avg" not in calibrators


class TestApplyCalibrators:
    def test_passthrough_for_uncalibrated_targets(self) -> None:
        raw_preds = {"avg": [0.250, 0.260, 0.270], "obp": [0.340, 0.350, 0.360]}
        calibrators: dict[str, object] = {}

        result = apply_calibrators(raw_preds, calibrators)

        assert result["avg"] == raw_preds["avg"]
        assert result["obp"] == raw_preds["obp"]

    def test_applies_calibration_to_matched_targets(self) -> None:
        model = _FakeModel(bias=-0.020)
        models = {"avg": model}
        n = 200
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        y_cal = {"avg": [0.250 + i * 0.001 for i in range(n)]}

        calibrators = fit_calibrators(models, X_cal, y_cal)

        new_preds = {"avg": [0.240, 0.280, 0.320], "obp": [0.340, 0.350, 0.360]}
        result = apply_calibrators(new_preds, calibrators)

        assert result["obp"] == new_preds["obp"]
        assert result["avg"] != new_preds["avg"]


class TestSaveLoadCalibrators:
    def test_round_trip_isotonic(self, tmp_path: Path) -> None:
        model = _FakeModel(bias=-0.020)
        models = {"avg": model}
        n = 200
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        y_cal = {"avg": [0.250 + i * 0.001 for i in range(n)]}

        calibrators = fit_calibrators(models, X_cal, y_cal, method="isotonic")

        path = tmp_path / "calibrators.joblib"
        save_calibrators(calibrators, path)
        loaded = load_calibrators(path)

        raw_preds = {"avg": [0.240, 0.280, 0.320]}
        result1 = apply_calibrators(raw_preds, calibrators)
        result2 = apply_calibrators(raw_preds, loaded)

        for i in range(len(raw_preds["avg"])):
            assert result1["avg"][i] == pytest.approx(result2["avg"][i])

    def test_round_trip_affine(self, tmp_path: Path) -> None:
        model = _FakeModel(bias=-0.020)
        models = {"avg": model}
        n = 200
        rng = np.random.default_rng(42)
        X_cal = rng.standard_normal((n, 5))
        y_cal = {"avg": [0.250 + i * 0.001 for i in range(n)]}

        calibrators = fit_calibrators(models, X_cal, y_cal, method="affine")

        path = tmp_path / "calibrators.joblib"
        save_calibrators(calibrators, path)
        loaded = load_calibrators(path)

        raw_preds = {"avg": [0.240, 0.280, 0.320]}
        result1 = apply_calibrators(raw_preds, calibrators)
        result2 = apply_calibrators(raw_preds, loaded)

        for i in range(len(raw_preds["avg"])):
            assert result1["avg"][i] == pytest.approx(result2["avg"][i])


class TestFitMultifoldCalibratorsAffine:
    def test_averages_per_fold_slopes_and_intercepts(self) -> None:
        rng = np.random.default_rng(42)
        n = 100
        slopes = [0.9, 1.1, 1.0]
        intercepts = [0.01, -0.01, 0.02]
        folds: list[FoldData] = []
        for slope, intercept in zip(slopes, intercepts, strict=True):
            preds = rng.uniform(0.200, 0.400, n)
            actuals = slope * preds + intercept
            folds.append({"era": (preds, actuals)})

        calibrators = fit_multifold_calibrators(folds, method="affine")

        assert "era" in calibrators
        cal = calibrators["era"]
        assert isinstance(cal, AffineCalibrator)
        expected_slope = sum(slopes) / len(slopes)
        expected_intercept = sum(intercepts) / len(intercepts)
        assert cal.slope == pytest.approx(expected_slope, abs=0.01)
        assert cal.intercept == pytest.approx(expected_intercept, abs=0.001)

    def test_skips_target_with_too_few_observations_in_all_folds(self) -> None:
        rng = np.random.default_rng(42)
        folds: list[FoldData] = []
        for _ in range(3):
            preds = rng.uniform(0.200, 0.400, 30)  # < 50 per fold
            actuals = preds + 0.01
            folds.append({"era": (preds, actuals)})

        calibrators = fit_multifold_calibrators(folds, method="affine")

        assert "era" not in calibrators

    def test_uses_only_folds_with_enough_observations(self) -> None:
        rng = np.random.default_rng(42)
        # Two qualifying folds with known slopes, one small fold
        slopes = [0.8, 1.2]
        intercepts = [0.05, -0.05]
        folds: list[FoldData] = []
        for slope, intercept in zip(slopes, intercepts, strict=True):
            preds = rng.uniform(0.200, 0.400, 100)
            actuals = slope * preds + intercept
            folds.append({"era": (preds, actuals)})
        # Small fold that should be skipped
        small_preds = rng.uniform(0.200, 0.400, 30)
        folds.append({"era": (small_preds, small_preds + 0.01)})

        calibrators = fit_multifold_calibrators(folds, method="affine")

        assert "era" in calibrators
        cal = calibrators["era"]
        assert isinstance(cal, AffineCalibrator)
        expected_slope = sum(slopes) / len(slopes)  # average of qualifying folds only
        expected_intercept = sum(intercepts) / len(intercepts)
        assert cal.slope == pytest.approx(expected_slope, abs=0.01)
        assert cal.intercept == pytest.approx(expected_intercept, abs=0.001)

    def test_multifold_more_stable_than_single_fold(self) -> None:
        rng = np.random.default_rng(42)
        true_slope = 1.0
        n = 80
        folds: list[FoldData] = []
        per_fold_slopes: list[float] = []
        for _ in range(5):
            preds = rng.uniform(0.200, 0.400, n)
            noise = rng.normal(0, 0.02, n)
            actuals = true_slope * preds + noise
            folds.append({"era": (preds, actuals)})
            # Fit per-fold slope for comparison
            fold_slope, _ = np.polyfit(preds, actuals, 1)
            per_fold_slopes.append(float(fold_slope))

        calibrators = fit_multifold_calibrators(folds, method="affine")
        cal = calibrators["era"]
        assert isinstance(cal, AffineCalibrator)

        # Aggregated slope should be closer to true than any individual fold
        agg_deviation = abs(cal.slope - true_slope)
        individual_deviations = [abs(s - true_slope) for s in per_fold_slopes]
        assert agg_deviation < max(individual_deviations)


class TestFitMultifoldCalibratorsIsotonic:
    def test_pools_all_fold_data(self) -> None:
        rng = np.random.default_rng(42)
        folds: list[FoldData] = []
        for _ in range(3):
            preds = rng.uniform(2.0, 5.0, 60)
            actuals = preds - 0.3  # systematic bias
            folds.append({"era": (preds, actuals)})

        calibrators = fit_multifold_calibrators(folds, method="isotonic")

        assert "era" in calibrators
        # Verify it can predict (i.e., it's a fitted IsotonicRegression)
        test_preds = np.array([3.0, 4.0, 5.0])
        result = calibrators["era"].predict(test_preds)
        assert len(result) == 3
        # Should correct predictions downward (since actuals = preds - 0.3)
        assert all(r < p for r, p in zip(result, test_preds, strict=True))

    def test_skips_target_with_too_few_pooled_observations(self) -> None:
        rng = np.random.default_rng(42)
        folds: list[FoldData] = []
        for _ in range(3):
            preds = rng.uniform(2.0, 5.0, 15)  # 15 * 3 = 45 < 50
            actuals = preds - 0.3
            folds.append({"era": (preds, actuals)})

        calibrators = fit_multifold_calibrators(folds, method="isotonic")

        assert "era" not in calibrators


class _FakeModel:
    """A fake model that returns predictions with a known bias."""

    def __init__(self, bias: float) -> None:
        self._bias = bias

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        n = X.shape[0]
        return np.linspace(0.250, 0.450, n) + self._bias
