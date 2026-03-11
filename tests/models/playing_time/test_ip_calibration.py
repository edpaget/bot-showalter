"""Unit tests for IP calibration via isotonic regression."""

from typing import Any

import pytest

from fantasy_baseball_manager.models.playing_time.ip_calibration import (
    IPCalibrator,
    calibrate_ip,
    calibrate_ip_batch,
    fit_ip_calibrator,
)


class TestFitIPCalibrator:
    def test_returns_ip_calibrator(self) -> None:
        predicted = [10.0, 20.0, 30.0, 40.0, 50.0]
        actual = [12.0, 22.0, 28.0, 38.0, 48.0]
        cal = fit_ip_calibrator(predicted, actual)
        assert isinstance(cal, IPCalibrator)

    def test_thresholds_are_sorted(self) -> None:
        predicted = [50.0, 10.0, 30.0, 20.0, 40.0]
        actual = [48.0, 12.0, 28.0, 22.0, 38.0]
        cal = fit_ip_calibrator(predicted, actual)
        assert list(cal.x_thresholds) == sorted(cal.x_thresholds)

    def test_corrects_systematic_upward_bias(self) -> None:
        # Model overpredicts by ~20 at each level
        predicted = [20.0, 40.0, 60.0, 80.0, 100.0]
        actual = [5.0, 20.0, 40.0, 60.0, 80.0]
        cal = fit_ip_calibrator(predicted, actual)
        # Calibrated value for 60 should be closer to 40 than to 60
        result = calibrate_ip(60.0, cal)
        assert result < 55.0
        assert result > 25.0

    def test_preserves_monotonicity(self) -> None:
        # Isotonic regression guarantees monotone output
        predicted = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        actual = [8.0, 15.0, 35.0, 30.0, 55.0, 58.0]  # non-monotone actuals
        cal = fit_ip_calibrator(predicted, actual)
        results = [calibrate_ip(x, cal) for x in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]

    def test_identity_when_predicted_equals_actual(self) -> None:
        predicted = [10.0, 20.0, 30.0, 40.0, 50.0]
        actual = [10.0, 20.0, 30.0, 40.0, 50.0]
        cal = fit_ip_calibrator(predicted, actual)
        for x in predicted:
            assert abs(calibrate_ip(x, cal) - x) < 1.0


class TestCalibrateIP:
    def test_interpolates_between_thresholds(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(10.0, 30.0, 50.0),
            y_calibrated=(5.0, 25.0, 45.0),
        )
        # Midpoint between 10 and 30 should give midpoint between 5 and 25
        result = calibrate_ip(20.0, cal)
        assert abs(result - 15.0) < 0.01

    def test_clips_below_minimum(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(10.0, 30.0, 50.0),
            y_calibrated=(5.0, 25.0, 45.0),
        )
        result = calibrate_ip(0.0, cal)
        assert result == pytest.approx(5.0)

    def test_clips_above_maximum(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(10.0, 30.0, 50.0),
            y_calibrated=(5.0, 25.0, 45.0),
        )
        result = calibrate_ip(100.0, cal)
        assert result == pytest.approx(45.0)

    def test_exact_threshold_value(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(10.0, 30.0, 50.0),
            y_calibrated=(5.0, 25.0, 45.0),
        )
        result = calibrate_ip(30.0, cal)
        assert result == pytest.approx(25.0)


def _make_prediction(player_id: int, player_type: str, ip_or_pa: float) -> dict[str, Any]:
    stat_key = "ip" if player_type == "pitcher" else "pa"
    return {
        "player_id": player_id,
        "season": 2024,
        "player_type": player_type,
        stat_key: ip_or_pa,
    }


class TestCalibrateIPBatch:
    def test_applies_calibration_to_pitcher_ip(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 100.0, 200.0),
            y_calibrated=(0.0, 80.0, 180.0),
        )
        preds = [_make_prediction(1, "pitcher", 100.0)]
        result = calibrate_ip_batch(preds, cal)
        assert result[0]["ip"] == pytest.approx(80.0)

    def test_preserves_batter_predictions(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 100.0, 200.0),
            y_calibrated=(0.0, 80.0, 180.0),
        )
        preds = [_make_prediction(1, "batter", 600.0)]
        result = calibrate_ip_batch(preds, cal)
        assert result[0]["pa"] == 600.0

    def test_filters_to_target_count(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 250.0),
            y_calibrated=(0.0, 250.0),  # identity
        )
        preds = [_make_prediction(i, "pitcher", float(i * 10)) for i in range(1, 11)]
        result = calibrate_ip_batch(preds, cal, target_pitcher_count=5)
        pitchers = [p for p in result if p["player_type"] == "pitcher"]
        assert len(pitchers) == 5
        # Should keep the top 5 by IP
        kept_ids = {p["player_id"] for p in pitchers}
        assert kept_ids == {6, 7, 8, 9, 10}

    def test_target_count_excludes_filtered_pitchers(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 250.0),
            y_calibrated=(0.0, 250.0),
        )
        preds = [_make_prediction(i, "pitcher", float(i * 10)) for i in range(1, 6)]
        result = calibrate_ip_batch(preds, cal, target_pitcher_count=3)
        # Filtered pitchers should not appear at all
        assert len(result) == 3

    def test_target_count_preserves_batters(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 250.0),
            y_calibrated=(0.0, 250.0),
        )
        preds = [
            _make_prediction(1, "batter", 600.0),
            _make_prediction(2, "pitcher", 100.0),
            _make_prediction(3, "pitcher", 50.0),
        ]
        result = calibrate_ip_batch(preds, cal, target_pitcher_count=1)
        assert len(result) == 2  # 1 batter + 1 pitcher
        types = {p["player_type"] for p in result}
        assert types == {"batter", "pitcher"}

    def test_no_calibrator_returns_unchanged(self) -> None:
        preds = [_make_prediction(1, "pitcher", 150.0)]
        result = calibrate_ip_batch(preds, None)
        assert result[0]["ip"] == 150.0

    def test_no_calibrator_still_filters_by_count(self) -> None:
        preds = [_make_prediction(i, "pitcher", float(i * 10)) for i in range(1, 6)]
        result = calibrate_ip_batch(preds, None, target_pitcher_count=3)
        assert len(result) == 3

    def test_empty_predictions(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 100.0),
            y_calibrated=(0.0, 80.0),
        )
        result = calibrate_ip_batch([], cal)
        assert result == []

    def test_target_count_greater_than_actual(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 250.0),
            y_calibrated=(0.0, 250.0),
        )
        preds = [_make_prediction(i, "pitcher", float(i * 10)) for i in range(1, 4)]
        result = calibrate_ip_batch(preds, cal, target_pitcher_count=10)
        assert len(result) == 3  # all kept

    def test_clamps_calibrated_values(self) -> None:
        # Calibrator that maps to values outside [0, 250]
        cal = IPCalibrator(
            x_thresholds=(0.0, 100.0),
            y_calibrated=(-10.0, 300.0),
        )
        preds = [
            _make_prediction(1, "pitcher", 0.0),
            _make_prediction(2, "pitcher", 100.0),
        ]
        result = calibrate_ip_batch(preds, cal)
        assert result[0]["ip"] >= 0.0
        assert result[1]["ip"] <= 250.0

    def test_does_not_mutate_input(self) -> None:
        cal = IPCalibrator(
            x_thresholds=(0.0, 100.0),
            y_calibrated=(0.0, 80.0),
        )
        preds = [_make_prediction(1, "pitcher", 100.0)]
        original_ip = preds[0]["ip"]
        calibrate_ip_batch(preds, cal)
        assert preds[0]["ip"] == original_ip
