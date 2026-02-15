from __future__ import annotations

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
            {"zone": 5, "description": "foul", "plate_x": 0.0, "plate_z": 2.5},
            # In-zone swing and miss (whiff)
            {"zone": 5, "description": "swinging_strike", "plate_x": 0.1, "plate_z": 2.6},
            # Out-of-zone swing (chase) and contact
            {"zone": 14, "description": "foul", "plate_x": 1.5, "plate_z": 3.8},
            # Out-of-zone take (ball)
            {"zone": 14, "description": "ball", "plate_x": 1.5, "plate_z": 3.8},
            # In-zone called strike (no swing)
            {"zone": 5, "description": "called_strike", "plate_x": 0.0, "plate_z": 2.5},
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
        assert all(v == pytest.approx(0.0) for v in result.values())
        assert len(result) == 5

    def test_all_in_zone_called_strikes(self) -> None:
        rows = [
            {"zone": 1, "description": "called_strike", "plate_x": 0.0, "plate_z": 3.0},
            {"zone": 2, "description": "called_strike", "plate_x": 0.3, "plate_z": 3.0},
        ]
        result = plate_discipline_profile(rows)
        assert result["chase_rate"] == pytest.approx(0.0)
        assert result["zone_contact_pct"] == pytest.approx(0.0)
        assert result["whiff_rate"] == pytest.approx(0.0)
        assert result["swinging_strike_pct"] == pytest.approx(0.0)
        assert result["called_strike_pct"] == pytest.approx(100.0)

    def test_no_swings(self) -> None:
        rows = [
            {"zone": 5, "description": "ball", "plate_x": 0.0, "plate_z": 2.5},
            {"zone": 14, "description": "ball", "plate_x": 1.5, "plate_z": 3.8},
        ]
        result = plate_discipline_profile(rows)
        assert result["chase_rate"] == pytest.approx(0.0)
        assert result["zone_contact_pct"] == pytest.approx(0.0)
        assert result["whiff_rate"] == pytest.approx(0.0)

    def test_all_chases(self) -> None:
        rows = [
            {"zone": 14, "description": "swinging_strike", "plate_x": 1.5, "plate_z": 3.8},
            {"zone": 13, "description": "foul", "plate_x": -1.5, "plate_z": 3.8},
        ]
        result = plate_discipline_profile(rows)
        assert result["chase_rate"] == pytest.approx(100.0)
        assert result["whiff_rate"] == pytest.approx(50.0)

    def test_none_zone_rows_skipped(self) -> None:
        rows = [
            {"zone": None, "description": "ball", "plate_x": 0.0, "plate_z": 2.5},
            {"zone": 5, "description": "called_strike", "plate_x": 0.0, "plate_z": 2.5},
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
            {"zone": 5, "description": "hit_into_play", "plate_x": 0.0, "plate_z": 2.5},
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

    def test_transform_callable(self) -> None:
        assert PLATE_DISCIPLINE.transform is plate_discipline_profile
