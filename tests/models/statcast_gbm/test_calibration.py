from typing import TYPE_CHECKING

import numpy as np
import pytest

from fantasy_baseball_manager.models.statcast_gbm.calibration import (
    AffineCalibrator,
    apply_calibrators,
    fit_calibrators,
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


class _FakeModel:
    """A fake model that returns predictions with a known bias."""

    def __init__(self, bias: float) -> None:
        self._bias = bias

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        n = X.shape[0]
        return np.linspace(0.250, 0.450, n) + self._bias
