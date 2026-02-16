from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.batted_ball import (
    BATTED_BALL,
    batted_ball_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestBattedBallProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 1},
            {"launch_speed": 90.0, "launch_angle": 10.0, "barrel": 0},
            {"launch_speed": 95.0, "launch_angle": 15.0, "barrel": 0},
            {"launch_speed": 105.0, "launch_angle": 30.0, "barrel": 1},
        ]
        result = batted_ball_profile(rows)
        assert result["avg_exit_velo"] == pytest.approx(97.5)
        assert result["max_exit_velo"] == pytest.approx(105.0)
        assert result["avg_launch_angle"] == pytest.approx(20.0)
        # barrel_pct: 2/4 = 50%
        assert result["barrel_pct"] == pytest.approx(50.0)
        # hard_hit_pct: launch_speed >= 95 → 3/4 = 75%
        assert result["hard_hit_pct"] == pytest.approx(75.0)

    def test_filters_to_batted_ball_events(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 1},
            {"launch_speed": None, "launch_angle": None, "barrel": None},
            {"launch_speed": 90.0, "launch_angle": 10.0, "barrel": 0},
        ]
        result = batted_ball_profile(rows)
        # Only 2 batted ball events
        assert result["avg_exit_velo"] == pytest.approx(95.0)
        assert result["max_exit_velo"] == pytest.approx(100.0)

    def test_empty_rows(self) -> None:
        result = batted_ball_profile([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 5

    def test_no_batted_ball_events(self) -> None:
        rows = [
            {"launch_speed": None, "launch_angle": None, "barrel": None},
            {"launch_speed": None, "launch_angle": None, "barrel": None},
        ]
        result = batted_ball_profile(rows)
        assert all(math.isnan(v) for v in result.values())

    def test_output_keys(self) -> None:
        result = batted_ball_profile([])
        expected_keys = {
            "avg_exit_velo",
            "max_exit_velo",
            "avg_launch_angle",
            "barrel_pct",
            "hard_hit_pct",
        }
        assert set(result.keys()) == expected_keys

    def test_all_hard_hit(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 1},
            {"launch_speed": 95.0, "launch_angle": 15.0, "barrel": 1},
        ]
        result = batted_ball_profile(rows)
        assert result["hard_hit_pct"] == pytest.approx(100.0)
        assert result["barrel_pct"] == pytest.approx(100.0)

    def test_non_one_barrel_not_counted(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 2},
            {"launch_speed": 95.0, "launch_angle": 15.0, "barrel": 1},
        ]
        result = batted_ball_profile(rows)
        assert result["barrel_pct"] == pytest.approx(50.0)

    def test_no_hard_hit(self) -> None:
        rows = [
            {"launch_speed": 80.0, "launch_angle": 5.0, "barrel": 0},
            {"launch_speed": 85.0, "launch_angle": 10.0, "barrel": 0},
        ]
        result = batted_ball_profile(rows)
        assert result["hard_hit_pct"] == pytest.approx(0.0)


class TestBattedBallTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(BATTED_BALL, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert BATTED_BALL.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(BATTED_BALL.outputs) == 5

    def test_transform_callable(self) -> None:
        assert BATTED_BALL.transform is batted_ball_profile
