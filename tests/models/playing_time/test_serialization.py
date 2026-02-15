from pathlib import Path

import pytest

from fantasy_baseball_manager.models.playing_time.aging import AgingCurve
from fantasy_baseball_manager.models.playing_time.engine import (
    PlayingTimeCoefficients,
    ResidualBuckets,
    ResidualPercentiles,
)
from fantasy_baseball_manager.models.playing_time.serialization import (
    load_aging_curves,
    load_coefficients,
    load_residual_buckets,
    save_aging_curves,
    save_coefficients,
    save_residual_buckets,
)


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        batter = PlayingTimeCoefficients(
            feature_names=("pa_1", "pa_2", "age"),
            coefficients=(0.5, 0.3, -1.0),
            intercept=100.0,
            r_squared=0.85,
            player_type="batter",
        )
        pitcher = PlayingTimeCoefficients(
            feature_names=("ip_1", "ip_2", "age"),
            coefficients=(0.4, 0.2, -0.5),
            intercept=50.0,
            r_squared=0.80,
            player_type="pitcher",
        )
        coefficients = {"batter": batter, "pitcher": pitcher}
        path = tmp_path / "pt_coefficients.joblib"
        save_coefficients(coefficients, path)
        loaded = load_coefficients(path)
        assert loaded["batter"] == batter
        assert loaded["pitcher"] == pitcher

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.joblib"
        with pytest.raises(FileNotFoundError):
            load_coefficients(path)


class TestAgingCurveSerialization:
    def test_save_load_aging_curves_roundtrip(self, tmp_path: Path) -> None:
        batter_curve = AgingCurve(peak_age=27.0, improvement_rate=0.01, decline_rate=0.005, player_type="batter")
        pitcher_curve = AgingCurve(peak_age=26.0, improvement_rate=0.008, decline_rate=0.007, player_type="pitcher")
        curves = {"batter": batter_curve, "pitcher": pitcher_curve}
        path = tmp_path / "aging_curves.joblib"
        save_aging_curves(curves, path)
        loaded = load_aging_curves(path)
        assert loaded["batter"] == batter_curve
        assert loaded["pitcher"] == pitcher_curve

    def test_load_missing_aging_curves_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.joblib"
        with pytest.raises(FileNotFoundError):
            load_aging_curves(path)


class TestResidualBucketsSerialization:
    def test_save_load_residual_buckets_roundtrip(self, tmp_path: Path) -> None:
        percs = ResidualPercentiles(
            p10=-30.0,
            p25=-10.0,
            p50=2.0,
            p75=15.0,
            p90=40.0,
            count=100,
            std=25.0,
            mean_offset=1.5,
        )
        batter = ResidualBuckets(buckets={"all": percs, "young_healthy": percs}, player_type="batter")
        pitcher = ResidualBuckets(buckets={"all": percs}, player_type="pitcher")
        data = {"batter": batter, "pitcher": pitcher}
        path = tmp_path / "pt_residual_buckets.joblib"
        save_residual_buckets(data, path)
        loaded = load_residual_buckets(path)
        assert loaded["batter"] == batter
        assert loaded["pitcher"] == pitcher

    def test_load_missing_residual_buckets_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.joblib"
        with pytest.raises(FileNotFoundError):
            load_residual_buckets(path)
