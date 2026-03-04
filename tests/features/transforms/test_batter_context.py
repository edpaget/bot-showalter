from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.batter_context import (
    BATTER_CONTEXT,
    batter_context_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestBatterContextProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            {"pfx_z": 10.0, "launch_angle": 30.0, "hc_x": 80.0, "hc_y": 150.0, "stand": "R"},
            {"pfx_z": 8.0, "launch_angle": 35.0, "hc_x": 170.0, "hc_y": 150.0, "stand": "L"},
            {"pfx_z": 12.0, "launch_angle": 5.0, "hc_x": 80.0, "hc_y": 150.0, "stand": "R"},
            {"pfx_z": 6.0, "launch_angle": 28.0, "hc_x": 125.42, "hc_y": 150.0, "stand": "R"},
        ]
        result = batter_context_profile(rows)
        # avg_vert_break_faced: mean(10, 8, 12, 6) = 9.0
        assert result["avg_vert_break_faced"] == pytest.approx(9.0)
        # Fly balls (25 <= angle < 50): rows 0 (R, hc_x=80 → pull), 1 (L, hc_x=170 → pull), 3 (R, center)
        # Pulled: 2 out of 3
        assert result["pull_fb_rate"] == pytest.approx(2.0 / 3.0)

    def test_empty_rows(self) -> None:
        result = batter_context_profile([])
        assert math.isnan(result["avg_vert_break_faced"])
        assert math.isnan(result["pull_fb_rate"])

    def test_null_pfx_z(self) -> None:
        rows = [
            {"pfx_z": 10.0, "launch_angle": 5.0, "hc_x": 80.0, "hc_y": 150.0, "stand": "R"},
            {"pfx_z": None, "launch_angle": 5.0, "hc_x": 80.0, "hc_y": 150.0, "stand": "R"},
            {"pfx_z": 6.0, "launch_angle": 5.0, "hc_x": 80.0, "hc_y": 150.0, "stand": "R"},
        ]
        result = batter_context_profile(rows)
        # Only non-null pfx_z: mean(10, 6) = 8.0
        assert result["avg_vert_break_faced"] == pytest.approx(8.0)

    def test_no_fly_balls(self) -> None:
        rows = [
            {"pfx_z": 10.0, "launch_angle": 5.0, "hc_x": 80.0, "hc_y": 150.0, "stand": "R"},
            {"pfx_z": 8.0, "launch_angle": 15.0, "hc_x": 80.0, "hc_y": 150.0, "stand": "R"},
        ]
        result = batter_context_profile(rows)
        assert result["avg_vert_break_faced"] == pytest.approx(9.0)
        assert math.isnan(result["pull_fb_rate"])

    def test_pull_right_handed(self) -> None:
        # R batter, hc_x far left → negative spray angle → pull
        rows = [
            {"pfx_z": 10.0, "launch_angle": 30.0, "hc_x": 60.0, "hc_y": 150.0, "stand": "R"},
        ]
        result = batter_context_profile(rows)
        assert result["pull_fb_rate"] == pytest.approx(1.0)

    def test_pull_left_handed(self) -> None:
        # L batter, hc_x far right → positive spray angle → pull
        rows = [
            {"pfx_z": 10.0, "launch_angle": 30.0, "hc_x": 190.0, "hc_y": 150.0, "stand": "L"},
        ]
        result = batter_context_profile(rows)
        assert result["pull_fb_rate"] == pytest.approx(1.0)


class TestBatterContextFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(BATTER_CONTEXT, TransformFeature)

    def test_source(self) -> None:
        assert BATTER_CONTEXT.source == Source.STATCAST

    def test_outputs(self) -> None:
        assert BATTER_CONTEXT.outputs == ("avg_vert_break_faced", "pull_fb_rate")
