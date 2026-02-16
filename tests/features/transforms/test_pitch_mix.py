from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.pitch_mix import (
    PITCH_MIX,
    pitch_mix_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestPitchMixProfile:
    def test_basic_usage_percentages(self) -> None:
        rows = [
            {"pitch_type": "FF", "release_speed": 95.0},
            {"pitch_type": "FF", "release_speed": 96.0},
            {"pitch_type": "SL", "release_speed": 85.0},
            {"pitch_type": "CH", "release_speed": 82.0},
        ]
        result = pitch_mix_profile(rows)
        assert result["ff_pct"] == pytest.approx(50.0)
        assert result["sl_pct"] == pytest.approx(25.0)
        assert result["ch_pct"] == pytest.approx(25.0)
        assert result["cu_pct"] == pytest.approx(0.0)
        assert result["si_pct"] == pytest.approx(0.0)
        assert result["fc_pct"] == pytest.approx(0.0)

    def test_average_velocities(self) -> None:
        rows = [
            {"pitch_type": "FF", "release_speed": 95.0},
            {"pitch_type": "FF", "release_speed": 97.0},
            {"pitch_type": "SL", "release_speed": 85.0},
        ]
        result = pitch_mix_profile(rows)
        assert result["ff_velo"] == pytest.approx(96.0)
        assert result["sl_velo"] == pytest.approx(85.0)

    def test_unused_pitch_type_nan_velo(self) -> None:
        rows = [
            {"pitch_type": "FF", "release_speed": 95.0},
        ]
        result = pitch_mix_profile(rows)
        assert math.isnan(result["sl_velo"])
        assert math.isnan(result["ch_velo"])

    def test_empty_rows(self) -> None:
        result = pitch_mix_profile([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 12

    def test_missing_pitch_type_ignored(self) -> None:
        rows = [
            {"pitch_type": None, "release_speed": 90.0},
            {"pitch_type": "FF", "release_speed": 95.0},
        ]
        result = pitch_mix_profile(rows)
        assert result["ff_pct"] == pytest.approx(100.0)

    def test_unknown_pitch_type_ignored(self) -> None:
        rows = [
            {"pitch_type": "KC", "release_speed": 78.0},
            {"pitch_type": "FF", "release_speed": 95.0},
        ]
        result = pitch_mix_profile(rows)
        assert result["ff_pct"] == pytest.approx(100.0)

    def test_output_keys(self) -> None:
        result = pitch_mix_profile([])
        expected_keys = {
            "ff_pct",
            "ff_velo",
            "sl_pct",
            "sl_velo",
            "ch_pct",
            "ch_velo",
            "cu_pct",
            "cu_velo",
            "si_pct",
            "si_velo",
            "fc_pct",
            "fc_velo",
        }
        assert set(result.keys()) == expected_keys

    def test_missing_release_speed(self) -> None:
        rows = [
            {"pitch_type": "FF", "release_speed": None},
            {"pitch_type": "FF", "release_speed": 95.0},
        ]
        result = pitch_mix_profile(rows)
        # FF counted for pct, but only the non-None speed contributes to velo
        assert result["ff_pct"] == pytest.approx(100.0)
        assert result["ff_velo"] == pytest.approx(95.0)


class TestPitchMixTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(PITCH_MIX, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert PITCH_MIX.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(PITCH_MIX.outputs) == 12

    def test_transform_callable(self) -> None:
        assert PITCH_MIX.transform is pitch_mix_profile
