from __future__ import annotations

import math
from typing import Any

import pytest

from fantasy_baseball_manager.features.transforms.batter_trends import (
    BATTER_STABILITY,
    BATTER_TRENDS,
    batter_stability_profile,
    batter_trend_profile,
)
from fantasy_baseball_manager.features.types import DerivedTransformFeature


class TestBatterTrendProfile:
    def test_positive_trend(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.300, "avg_2": 0.280, "obp_1": 0.380, "obp_2": 0.360, "slg_1": 0.500, "slg_2": 0.470}
        ]
        result = batter_trend_profile(rows)
        assert result["avg_trend"] == pytest.approx(0.020)
        assert result["obp_trend"] == pytest.approx(0.020)
        assert result["slg_trend"] == pytest.approx(0.030)

    def test_negative_trend(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.250, "avg_2": 0.290, "obp_1": 0.330, "obp_2": 0.370, "slg_1": 0.400, "slg_2": 0.450}
        ]
        result = batter_trend_profile(rows)
        assert result["avg_trend"] == pytest.approx(-0.040)
        assert result["obp_trend"] == pytest.approx(-0.040)
        assert result["slg_trend"] == pytest.approx(-0.050)

    def test_null_lag2_produces_nan(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.300, "avg_2": None, "obp_1": 0.380, "obp_2": None, "slg_1": 0.500, "slg_2": None}
        ]
        result = batter_trend_profile(rows)
        assert math.isnan(result["avg_trend"])
        assert math.isnan(result["obp_trend"])
        assert math.isnan(result["slg_trend"])

    def test_null_lag1_produces_nan(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": None, "avg_2": 0.280, "obp_1": None, "obp_2": 0.360, "slg_1": None, "slg_2": 0.470}
        ]
        result = batter_trend_profile(rows)
        assert math.isnan(result["avg_trend"])
        assert math.isnan(result["obp_trend"])
        assert math.isnan(result["slg_trend"])

    def test_zero_values(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.0, "avg_2": 0.0, "obp_1": 0.0, "obp_2": 0.0, "slg_1": 0.0, "slg_2": 0.0}
        ]
        result = batter_trend_profile(rows)
        assert result["avg_trend"] == pytest.approx(0.0)
        assert result["obp_trend"] == pytest.approx(0.0)
        assert result["slg_trend"] == pytest.approx(0.0)

    def test_nan_input_produces_nan(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "avg_1": float("nan"),
                "avg_2": 0.280,
                "obp_1": 0.380,
                "obp_2": float("nan"),
                "slg_1": 0.500,
                "slg_2": 0.470,
            }
        ]
        result = batter_trend_profile(rows)
        assert math.isnan(result["avg_trend"])
        assert math.isnan(result["obp_trend"])
        assert result["slg_trend"] == pytest.approx(0.030)


class TestBatterTrendsFeature:
    def test_is_derived_transform_feature(self) -> None:
        assert isinstance(BATTER_TRENDS, DerivedTransformFeature)

    def test_outputs(self) -> None:
        assert BATTER_TRENDS.outputs == ("avg_trend", "obp_trend", "slg_trend")

    def test_inputs(self) -> None:
        assert BATTER_TRENDS.inputs == ("avg_1", "avg_2", "obp_1", "obp_2", "slg_1", "slg_2")

    def test_transform_callable(self) -> None:
        assert BATTER_TRENDS.transform is batter_trend_profile


class TestBatterStabilityProfile:
    def test_stable_player(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.300, "avg_2": 0.295, "obp_1": 0.380, "obp_2": 0.375, "slg_1": 0.500, "slg_2": 0.495}
        ]
        result = batter_stability_profile(rows)
        assert result["avg_stability"] == pytest.approx(0.005)
        assert result["obp_stability"] == pytest.approx(0.005)
        assert result["slg_stability"] == pytest.approx(0.005)

    def test_volatile_player(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.250, "avg_2": 0.300, "obp_1": 0.320, "obp_2": 0.380, "slg_1": 0.400, "slg_2": 0.500}
        ]
        result = batter_stability_profile(rows)
        assert result["avg_stability"] == pytest.approx(0.050)
        assert result["obp_stability"] == pytest.approx(0.060)
        assert result["slg_stability"] == pytest.approx(0.100)

    def test_null_handling(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.300, "avg_2": None, "obp_1": 0.380, "obp_2": None, "slg_1": 0.500, "slg_2": None}
        ]
        result = batter_stability_profile(rows)
        assert math.isnan(result["avg_stability"])
        assert math.isnan(result["obp_stability"])
        assert math.isnan(result["slg_stability"])

    def test_zero_values(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.0, "avg_2": 0.0, "obp_1": 0.0, "obp_2": 0.0, "slg_1": 0.0, "slg_2": 0.0}
        ]
        result = batter_stability_profile(rows)
        assert result["avg_stability"] == pytest.approx(0.0)
        assert result["obp_stability"] == pytest.approx(0.0)
        assert result["slg_stability"] == pytest.approx(0.0)

    def test_stability_always_positive(self) -> None:
        rows: list[dict[str, Any]] = [
            {"avg_1": 0.250, "avg_2": 0.300, "obp_1": 0.320, "obp_2": 0.380, "slg_1": 0.400, "slg_2": 0.500}
        ]
        result = batter_stability_profile(rows)
        assert result["avg_stability"] >= 0
        assert result["obp_stability"] >= 0
        assert result["slg_stability"] >= 0


class TestBatterStabilityFeature:
    def test_is_derived_transform_feature(self) -> None:
        assert isinstance(BATTER_STABILITY, DerivedTransformFeature)

    def test_outputs(self) -> None:
        assert BATTER_STABILITY.outputs == ("avg_stability", "obp_stability", "slg_stability")

    def test_inputs(self) -> None:
        assert BATTER_STABILITY.inputs == ("avg_1", "avg_2", "obp_1", "obp_2", "slg_1", "slg_2")

    def test_transform_callable(self) -> None:
        assert BATTER_STABILITY.transform is batter_stability_profile
