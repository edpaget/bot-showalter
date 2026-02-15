from __future__ import annotations

import pytest

from fantasy_baseball_manager.features.transforms.expected_stats import (
    EXPECTED_STATS,
    expected_stats_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestExpectedStatsProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            {"estimated_ba_using_speedangle": 0.500, "estimated_woba_using_speedangle": 0.800},
            {"estimated_ba_using_speedangle": 0.300, "estimated_woba_using_speedangle": 0.400},
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.400)
        assert result["xwoba"] == pytest.approx(0.600)
        # xslg estimated from xwoba via scaling factor (xwoba / 0.320 * 0.400 approx)
        assert "xslg" in result
        assert result["xslg"] > 0.0

    def test_empty_rows(self) -> None:
        result = expected_stats_profile([])
        assert all(v == pytest.approx(0.0) for v in result.values())
        assert len(result) == 3

    def test_filters_null_xba(self) -> None:
        rows = [
            {"estimated_ba_using_speedangle": None, "estimated_woba_using_speedangle": None},
            {"estimated_ba_using_speedangle": 0.300, "estimated_woba_using_speedangle": 0.400},
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.300)
        assert result["xwoba"] == pytest.approx(0.400)

    def test_all_null(self) -> None:
        rows = [
            {"estimated_ba_using_speedangle": None, "estimated_woba_using_speedangle": None},
        ]
        result = expected_stats_profile(rows)
        assert all(v == pytest.approx(0.0) for v in result.values())

    def test_single_row(self) -> None:
        rows = [
            {"estimated_ba_using_speedangle": 0.250, "estimated_woba_using_speedangle": 0.350},
        ]
        result = expected_stats_profile(rows)
        assert result["xba"] == pytest.approx(0.250)
        assert result["xwoba"] == pytest.approx(0.350)

    def test_output_keys(self) -> None:
        result = expected_stats_profile([])
        expected_keys = {"xba", "xwoba", "xslg"}
        assert set(result.keys()) == expected_keys

    def test_xslg_scales_from_xwoba(self) -> None:
        rows = [
            {"estimated_ba_using_speedangle": 0.300, "estimated_woba_using_speedangle": 0.400},
        ]
        result = expected_stats_profile(rows)
        # xSLG is estimated as xwOBA * (league_slg / league_woba)
        # Using typical ratio of ~1.25 (0.400 / 0.320)
        assert result["xslg"] == pytest.approx(0.400 * 1.25)


class TestExpectedStatsTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(EXPECTED_STATS, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert EXPECTED_STATS.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(EXPECTED_STATS.outputs) == 3

    def test_transform_callable(self) -> None:
        assert EXPECTED_STATS.transform is expected_stats_profile
