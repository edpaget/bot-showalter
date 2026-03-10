from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.expected_stats import (
    EXPECTED_STATS,
    EXPECTED_STATS_ADVANCED,
    expected_stats_advanced_profile,
    expected_stats_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestExpectedStatsProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            {
                "estimated_ba_using_speedangle": 0.500,
                "estimated_woba_using_speedangle": 0.800,
                "estimated_slg_using_speedangle": 0.900,
            },
            {
                "estimated_ba_using_speedangle": 0.300,
                "estimated_woba_using_speedangle": 0.400,
                "estimated_slg_using_speedangle": 0.500,
            },
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.400)
        assert result["xwoba"] == pytest.approx(0.600)
        assert result["xslg"] == pytest.approx(0.700)

    def test_empty_rows(self) -> None:
        result = expected_stats_profile([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 3

    def test_filters_null_xba(self) -> None:
        rows = [
            {
                "estimated_ba_using_speedangle": None,
                "estimated_woba_using_speedangle": None,
                "estimated_slg_using_speedangle": None,
            },
            {
                "estimated_ba_using_speedangle": 0.300,
                "estimated_woba_using_speedangle": 0.400,
                "estimated_slg_using_speedangle": 0.500,
            },
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.300)
        assert result["xwoba"] == pytest.approx(0.400)
        assert result["xslg"] == pytest.approx(0.500)

    def test_all_null(self) -> None:
        rows = [
            {
                "estimated_ba_using_speedangle": None,
                "estimated_woba_using_speedangle": None,
                "estimated_slg_using_speedangle": None,
            },
        ]
        result = expected_stats_profile(rows)
        assert all(math.isnan(v) for v in result.values())

    def test_single_row(self) -> None:
        rows = [
            {
                "estimated_ba_using_speedangle": 0.250,
                "estimated_woba_using_speedangle": 0.350,
                "estimated_slg_using_speedangle": 0.450,
            },
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.250)
        assert result["xwoba"] == pytest.approx(0.350)
        assert result["xslg"] == pytest.approx(0.450)

    def test_output_keys(self) -> None:
        result = expected_stats_profile([])
        expected_keys = {"xba", "xwoba", "xslg"}
        assert set(result.keys()) == expected_keys

    def test_xslg_from_statcast_column(self) -> None:
        rows = [
            {
                "estimated_ba_using_speedangle": 0.300,
                "estimated_woba_using_speedangle": 0.400,
                "estimated_slg_using_speedangle": 0.550,
            },
            {
                "estimated_ba_using_speedangle": 0.250,
                "estimated_woba_using_speedangle": 0.350,
                "estimated_slg_using_speedangle": 0.450,
            },
        ]
        result = expected_stats_profile(rows)
        assert result["xslg"] == pytest.approx(0.500)

    def test_null_xslg_excluded_from_mean(self) -> None:
        rows = [
            {
                "estimated_ba_using_speedangle": 0.300,
                "estimated_woba_using_speedangle": 0.400,
                "estimated_slg_using_speedangle": 0.600,
            },
            {
                "estimated_ba_using_speedangle": 0.250,
                "estimated_woba_using_speedangle": 0.350,
                "estimated_slg_using_speedangle": None,
            },
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.275)
        assert result["xwoba"] == pytest.approx(0.375)
        assert result["xslg"] == pytest.approx(0.600)

    def test_all_null_xslg_returns_nan(self) -> None:
        rows = [
            {
                "estimated_ba_using_speedangle": 0.300,
                "estimated_woba_using_speedangle": 0.400,
                "estimated_slg_using_speedangle": None,
            },
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.300)
        assert result["xwoba"] == pytest.approx(0.400)
        assert math.isnan(result["xslg"])


class TestExpectedStatsTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(EXPECTED_STATS, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert EXPECTED_STATS.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(EXPECTED_STATS.outputs) == 3

    def test_transform_callable(self) -> None:
        assert EXPECTED_STATS.transform is expected_stats_profile


class TestExpectedStatsAdvancedProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            {"estimated_woba_using_speedangle": 1.200, "pitch_type": "FF"},
            {"estimated_woba_using_speedangle": 0.400, "pitch_type": "FF"},
            {"estimated_woba_using_speedangle": 0.300, "pitch_type": "SL"},
            {"estimated_woba_using_speedangle": 0.800, "pitch_type": "SI"},
        ]
        result = expected_stats_advanced_profile(rows)
        # elite: 1 out of 4 = 0.25
        assert result["xwoba_elite_rate"] == pytest.approx(0.25)
        # fastball (FF, SI): (1.2 + 0.4 + 0.8) / 3 = 0.8
        assert result["xwoba_fastball"] == pytest.approx(0.800)

    def test_empty_rows(self) -> None:
        result = expected_stats_advanced_profile([])
        assert math.isnan(result["xwoba_elite_rate"])
        assert math.isnan(result["xwoba_fastball"])

    def test_all_null_xwoba(self) -> None:
        rows = [{"estimated_woba_using_speedangle": None, "pitch_type": "FF"}]
        result = expected_stats_advanced_profile(rows)
        assert math.isnan(result["xwoba_elite_rate"])
        assert math.isnan(result["xwoba_fastball"])

    def test_no_fastballs(self) -> None:
        rows = [
            {"estimated_woba_using_speedangle": 0.500, "pitch_type": "SL"},
            {"estimated_woba_using_speedangle": 1.100, "pitch_type": "CU"},
        ]
        result = expected_stats_advanced_profile(rows)
        assert result["xwoba_elite_rate"] == pytest.approx(0.5)
        assert math.isnan(result["xwoba_fastball"])

    def test_all_elite(self) -> None:
        rows = [
            {"estimated_woba_using_speedangle": 1.500, "pitch_type": "FF"},
            {"estimated_woba_using_speedangle": 1.000, "pitch_type": "SI"},
        ]
        result = expected_stats_advanced_profile(rows)
        assert result["xwoba_elite_rate"] == pytest.approx(1.0)

    def test_no_elite(self) -> None:
        rows = [
            {"estimated_woba_using_speedangle": 0.200, "pitch_type": "FF"},
            {"estimated_woba_using_speedangle": 0.999, "pitch_type": "SL"},
        ]
        result = expected_stats_advanced_profile(rows)
        assert result["xwoba_elite_rate"] == pytest.approx(0.0)

    def test_output_keys(self) -> None:
        result = expected_stats_advanced_profile([])
        assert set(result.keys()) == {"xwoba_elite_rate", "xwoba_fastball"}

    def test_null_xwoba_excluded(self) -> None:
        rows = [
            {"estimated_woba_using_speedangle": None, "pitch_type": "FF"},
            {"estimated_woba_using_speedangle": 1.200, "pitch_type": "FF"},
        ]
        result = expected_stats_advanced_profile(rows)
        assert result["xwoba_elite_rate"] == pytest.approx(1.0)
        assert result["xwoba_fastball"] == pytest.approx(1.200)


class TestExpectedStatsAdvancedTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(EXPECTED_STATS_ADVANCED, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert EXPECTED_STATS_ADVANCED.source == Source.STATCAST

    def test_outputs(self) -> None:
        assert EXPECTED_STATS_ADVANCED.outputs == ("xwoba_elite_rate", "xwoba_fastball")

    def test_columns_include_pitch_type(self) -> None:
        assert "pitch_type" in EXPECTED_STATS_ADVANCED.columns

    def test_transform_callable(self) -> None:
        assert EXPECTED_STATS_ADVANCED.transform is expected_stats_advanced_profile
