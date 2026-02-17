from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.command import (
    COMMAND,
    command_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestCommandProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            # In-zone, first pitch, strike (called_strike)
            {"zone": 5, "description": "called_strike", "pitch_number": 1},
            # Out-of-zone, not first pitch
            {"zone": 14, "description": "ball", "pitch_number": 2},
            # In-zone, first pitch, not a strike (ball)
            {"zone": 3, "description": "ball", "pitch_number": 1},
            # Out-of-zone, first pitch, strike (swinging_strike)
            {"zone": 13, "description": "swinging_strike", "pitch_number": 1},
            # In-zone, not first pitch
            {"zone": 7, "description": "foul", "pitch_number": 3},
            # In-zone, not first pitch
            {"zone": 1, "description": "hit_into_play", "pitch_number": 2},
        ]
        result = command_profile(rows)
        # zone_rate: 4 in-zone (5,3,7,1) / 6 total with zone data = 66.67%
        assert result["zone_rate"] == pytest.approx(4 / 6 * 100.0)
        # first_pitch_strike_pct: 2 strikes (called_strike, swinging_strike) / 3 first pitches = 66.67%
        assert result["first_pitch_strike_pct"] == pytest.approx(2 / 3 * 100.0)

    def test_empty_rows(self) -> None:
        result = command_profile([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 2

    def test_all_in_zone(self) -> None:
        rows = [
            {"zone": 1, "description": "called_strike", "pitch_number": 2},
            {"zone": 5, "description": "ball", "pitch_number": 3},
            {"zone": 9, "description": "foul", "pitch_number": 1},
        ]
        result = command_profile(rows)
        assert result["zone_rate"] == pytest.approx(100.0)

    def test_no_first_pitches(self) -> None:
        rows = [
            {"zone": 5, "description": "called_strike", "pitch_number": 2},
            {"zone": 14, "description": "ball", "pitch_number": 3},
        ]
        result = command_profile(rows)
        assert math.isnan(result["first_pitch_strike_pct"])
        assert result["zone_rate"] == pytest.approx(50.0)

    def test_null_zone_excluded_from_zone_rate(self) -> None:
        rows = [
            {"zone": None, "description": "ball", "pitch_number": 2},
            {"zone": 5, "description": "called_strike", "pitch_number": 3},
        ]
        result = command_profile(rows)
        # Only 1 row with zone data, zone 5 is in-zone → 100%
        assert result["zone_rate"] == pytest.approx(100.0)

    def test_first_pitch_with_null_zone(self) -> None:
        rows = [
            {"zone": None, "description": "called_strike", "pitch_number": 1},
            {"zone": 5, "description": "ball", "pitch_number": 2},
        ]
        result = command_profile(rows)
        # First pitch has null zone but valid strike description → counts
        assert result["first_pitch_strike_pct"] == pytest.approx(100.0)
        # zone_rate only counts the row with zone data
        assert result["zone_rate"] == pytest.approx(100.0)

    def test_output_keys(self) -> None:
        result = command_profile([])
        assert set(result.keys()) == {"zone_rate", "first_pitch_strike_pct"}


class TestCommandTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(COMMAND, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert COMMAND.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(COMMAND.outputs) == 2

    def test_transform_callable(self) -> None:
        assert COMMAND.transform is command_profile

    def test_name(self) -> None:
        assert COMMAND.name == "command"
