from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.plate_discipline import (
    PLATE_DISCIPLINE,
    plate_discipline_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestPlateDisciplineProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            # In-zone swing and contact
            {"zone": 5, "description": "foul"},
            # In-zone swing and miss (whiff)
            {"zone": 5, "description": "swinging_strike"},
            # Out-of-zone swing (chase) and contact
            {"zone": 14, "description": "foul"},
            # Out-of-zone take (ball)
            {"zone": 14, "description": "ball"},
            # In-zone called strike (no swing)
            {"zone": 5, "description": "called_strike"},
        ]
        result = plate_discipline_profile(rows)
        # 5 total pitches
        # chase: 1 swing outside zone / 2 pitches outside zone = 50%
        assert result["chase_rate"] == pytest.approx(50.0)
        # zone_contact: 1 contact in zone / 2 swings in zone = 50%
        assert result["zone_contact_pct"] == pytest.approx(50.0)
        # whiff: 1 swing-and-miss / 3 swings total = 33.33...%
        assert result["whiff_rate"] == pytest.approx(100.0 / 3.0)
        # swinging_strike: 1 swing-and-miss / 5 total pitches = 20%
        assert result["swinging_strike_pct"] == pytest.approx(20.0)
        # called_strike: 1 / 5 total pitches = 20%
        assert result["called_strike_pct"] == pytest.approx(20.0)

    def test_empty_rows(self) -> None:
        result = plate_discipline_profile([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 5

    def test_all_in_zone_called_strikes(self) -> None:
        rows = [
            {"zone": 1, "description": "called_strike"},
            {"zone": 2, "description": "called_strike"},
        ]
        result = plate_discipline_profile(rows)
        assert math.isnan(result["chase_rate"])
        assert math.isnan(result["zone_contact_pct"])
        assert math.isnan(result["whiff_rate"])
        assert result["swinging_strike_pct"] == pytest.approx(0.0)
        assert result["called_strike_pct"] == pytest.approx(100.0)

    def test_no_swings(self) -> None:
        rows = [
            {"zone": 5, "description": "ball"},
            {"zone": 14, "description": "ball"},
        ]
        result = plate_discipline_profile(rows)
        assert result["chase_rate"] == pytest.approx(0.0)
        assert math.isnan(result["zone_contact_pct"])
        assert math.isnan(result["whiff_rate"])

    def test_all_chases(self) -> None:
        rows = [
            {"zone": 14, "description": "swinging_strike"},
            {"zone": 13, "description": "foul"},
        ]
        result = plate_discipline_profile(rows)
        assert result["chase_rate"] == pytest.approx(100.0)
        assert result["whiff_rate"] == pytest.approx(50.0)

    def test_none_zone_rows_skipped(self) -> None:
        rows = [
            {"zone": None, "description": "ball"},
            {"zone": 5, "description": "called_strike"},
        ]
        result = plate_discipline_profile(rows)
        # Only one valid row counted
        assert result["called_strike_pct"] == pytest.approx(100.0)

    def test_output_keys(self) -> None:
        result = plate_discipline_profile([])
        expected_keys = {
            "chase_rate",
            "zone_contact_pct",
            "whiff_rate",
            "swinging_strike_pct",
            "called_strike_pct",
        }
        assert set(result.keys()) == expected_keys

    def test_hit_into_play_is_contact(self) -> None:
        rows = [
            {"zone": 5, "description": "hit_into_play"},
        ]
        result = plate_discipline_profile(rows)
        assert result["zone_contact_pct"] == pytest.approx(100.0)
        assert result["whiff_rate"] == pytest.approx(0.0)


class TestPlateDisciplineTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(PLATE_DISCIPLINE, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert PLATE_DISCIPLINE.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(PLATE_DISCIPLINE.outputs) == 5

    def test_columns_exclude_unused(self) -> None:
        assert PLATE_DISCIPLINE.columns == ("zone", "description")

    def test_transform_callable(self) -> None:
        assert PLATE_DISCIPLINE.transform is plate_discipline_profile
