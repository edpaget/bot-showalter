from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.spin_profile import (
    SPIN_PROFILE,
    spin_profile_metrics,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestSpinProfileMetrics:
    def test_basic_metrics(self) -> None:
        rows = [
            {"release_spin_rate": 2400.0, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0, "release_extension": 6.5},
            {"release_spin_rate": 2600.0, "pitch_type": "FF", "pfx_x": -6.0, "pfx_z": 13.0, "release_extension": 6.3},
            {"release_spin_rate": 2800.0, "pitch_type": "SL", "pfx_x": 3.0, "pfx_z": -2.0, "release_extension": 6.1},
            {"release_spin_rate": 2900.0, "pitch_type": "CU", "pfx_x": 4.0, "pfx_z": -8.0, "release_extension": 6.0},
        ]
        result = spin_profile_metrics(rows)
        assert result["avg_spin_rate"] == pytest.approx(2675.0)
        assert result["ff_spin"] == pytest.approx(2500.0)
        assert result["sl_spin"] == pytest.approx(2800.0)
        assert result["cu_spin"] == pytest.approx(2900.0)
        assert result["avg_h_break"] == pytest.approx(-1.0)  # mean of -5, -6, 3, 4
        assert result["avg_v_break"] == pytest.approx(3.75)  # mean of 12, 13, -2, -8
        assert result["avg_extension"] == pytest.approx(6.225)  # mean of 6.5, 6.3, 6.1, 6.0
        assert result["ff_extension"] == pytest.approx(6.4)  # mean of 6.5, 6.3

    def test_per_pitch_type_break(self) -> None:
        rows = [
            {"release_spin_rate": 2400.0, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0},
            {"release_spin_rate": 2600.0, "pitch_type": "FF", "pfx_x": -6.0, "pfx_z": 13.0},
            {"release_spin_rate": 2800.0, "pitch_type": "SL", "pfx_x": 3.0, "pfx_z": -2.0},
            {"release_spin_rate": 2900.0, "pitch_type": "CU", "pfx_x": 4.0, "pfx_z": -8.0},
        ]
        result = spin_profile_metrics(rows)
        assert result["ff_h_break"] == pytest.approx(-5.5)
        assert result["ff_v_break"] == pytest.approx(12.5)
        assert result["sl_h_break"] == pytest.approx(3.0)
        assert result["sl_v_break"] == pytest.approx(-2.0)
        assert result["cu_h_break"] == pytest.approx(4.0)
        assert result["cu_v_break"] == pytest.approx(-8.0)
        assert math.isnan(result["ch_h_break"])
        assert math.isnan(result["ch_v_break"])

    def test_ch_break_metrics(self) -> None:
        rows = [
            {"release_spin_rate": 1800.0, "pitch_type": "CH", "pfx_x": -7.0, "pfx_z": 5.0},
            {"release_spin_rate": 1850.0, "pitch_type": "CH", "pfx_x": -9.0, "pfx_z": 3.0},
        ]
        result = spin_profile_metrics(rows)
        assert result["ch_h_break"] == pytest.approx(-8.0)
        assert result["ch_v_break"] == pytest.approx(4.0)

    def test_per_pitch_break_with_missing_pfx(self) -> None:
        rows = [
            {"release_spin_rate": 2400.0, "pitch_type": "FF", "pfx_x": None, "pfx_z": None},
            {"release_spin_rate": 2600.0, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0},
        ]
        result = spin_profile_metrics(rows)
        assert result["ff_h_break"] == pytest.approx(-5.0)
        assert result["ff_v_break"] == pytest.approx(12.0)

    def test_empty_rows(self) -> None:
        result = spin_profile_metrics([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 17

    def test_missing_spin_rate_rows_filtered(self) -> None:
        rows = [
            {"release_spin_rate": None, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0},
            {"release_spin_rate": 2400.0, "pitch_type": "FF", "pfx_x": -6.0, "pfx_z": 13.0},
        ]
        result = spin_profile_metrics(rows)
        assert result["avg_spin_rate"] == pytest.approx(2400.0)
        assert result["ff_spin"] == pytest.approx(2400.0)

    def test_unknown_pitch_type_included_in_avg_only(self) -> None:
        rows = [
            {"release_spin_rate": 2400.0, "pitch_type": "KC", "pfx_x": 1.0, "pfx_z": -5.0},
            {"release_spin_rate": 2600.0, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0},
        ]
        result = spin_profile_metrics(rows)
        # Both contribute to avg_spin_rate
        assert result["avg_spin_rate"] == pytest.approx(2500.0)
        # Only FF has pitch-specific spin
        assert result["ff_spin"] == pytest.approx(2600.0)
        assert math.isnan(result["sl_spin"])

    def test_no_ff_pitches(self) -> None:
        rows = [
            {"release_spin_rate": 2800.0, "pitch_type": "SL", "pfx_x": 3.0, "pfx_z": -2.0},
        ]
        result = spin_profile_metrics(rows)
        assert math.isnan(result["ff_spin"])
        assert result["sl_spin"] == pytest.approx(2800.0)

    def test_break_with_missing_pfx(self) -> None:
        rows = [
            {"release_spin_rate": 2400.0, "pitch_type": "FF", "pfx_x": None, "pfx_z": None},
            {"release_spin_rate": 2600.0, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0},
        ]
        result = spin_profile_metrics(rows)
        # Only the non-null pfx row contributes to break averages
        assert result["avg_h_break"] == pytest.approx(-5.0)
        assert result["avg_v_break"] == pytest.approx(12.0)

    def test_output_keys(self) -> None:
        result = spin_profile_metrics([])
        expected_keys = {
            "avg_spin_rate",
            "ff_spin",
            "sl_spin",
            "cu_spin",
            "ch_spin",
            "avg_h_break",
            "avg_v_break",
            "ff_h_break",
            "ff_v_break",
            "sl_h_break",
            "sl_v_break",
            "cu_h_break",
            "cu_v_break",
            "ch_h_break",
            "ch_v_break",
            "avg_extension",
            "ff_extension",
        }
        assert set(result.keys()) == expected_keys

    def test_extension_null_excluded(self) -> None:
        rows = [
            {"release_spin_rate": 2400.0, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0, "release_extension": None},
            {"release_spin_rate": 2600.0, "pitch_type": "FF", "pfx_x": -6.0, "pfx_z": 13.0, "release_extension": 6.5},
        ]
        result = spin_profile_metrics(rows)
        assert result["avg_extension"] == pytest.approx(6.5)
        assert result["ff_extension"] == pytest.approx(6.5)

    def test_ff_extension_only_ff_pitches(self) -> None:
        rows = [
            {"release_spin_rate": 2400.0, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0, "release_extension": 6.5},
            {"release_spin_rate": 2800.0, "pitch_type": "SL", "pfx_x": 3.0, "pfx_z": -2.0, "release_extension": 5.8},
        ]
        result = spin_profile_metrics(rows)
        assert result["avg_extension"] == pytest.approx(6.15)  # mean of 6.5, 5.8
        assert result["ff_extension"] == pytest.approx(6.5)  # only FF pitch

    def test_all_null_spin_rates(self) -> None:
        rows = [
            {"release_spin_rate": None, "pitch_type": "FF", "pfx_x": -5.0, "pfx_z": 12.0},
        ]
        result = spin_profile_metrics(rows)
        assert math.isnan(result["avg_spin_rate"])
        assert math.isnan(result["ff_spin"])


class TestSpinProfileTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(SPIN_PROFILE, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert SPIN_PROFILE.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(SPIN_PROFILE.outputs) == 17

    def test_transform_callable(self) -> None:
        assert SPIN_PROFILE.transform is spin_profile_metrics
