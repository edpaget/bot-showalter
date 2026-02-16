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

    def test_batted_ball_extensions(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 1},
            {"launch_speed": 90.0, "launch_angle": 10.0, "barrel": 0},
            {"launch_speed": 95.0, "launch_angle": 15.0, "barrel": 0},
            {"launch_speed": 105.0, "launch_angle": 30.0, "barrel": 1},
        ]
        result = batted_ball_profile(rows)
        # gb_pct: angle < 10 → 0/4
        assert result["gb_pct"] == pytest.approx(0.0)
        # fb_pct: 25 <= angle < 50 → 2/4 (25, 30)
        assert result["fb_pct"] == pytest.approx(50.0)
        # ld_pct: 10 <= angle < 25 → 2/4 (10, 15)
        assert result["ld_pct"] == pytest.approx(50.0)
        # sweet_spot_pct: 8 <= angle <= 32 → 4/4
        assert result["sweet_spot_pct"] == pytest.approx(100.0)
        # exit_velo_p90: sorted [90, 95, 100, 105] → 90th percentile
        assert result["exit_velo_p90"] == pytest.approx(103.5)

    def test_ground_ball_heavy_profile(self) -> None:
        rows = [
            {"launch_speed": 95.0, "launch_angle": -5.0, "barrel": 0},
            {"launch_speed": 88.0, "launch_angle": 3.0, "barrel": 0},
            {"launch_speed": 102.0, "launch_angle": 8.0, "barrel": 0},
            {"launch_speed": 100.0, "launch_angle": 28.0, "barrel": 1},
        ]
        result = batted_ball_profile(rows)
        # gb_pct: angle < 10 → 3/4 (-5, 3, 8)
        assert result["gb_pct"] == pytest.approx(75.0)
        # fb_pct: 25 <= angle < 50 → 1/4 (28)
        assert result["fb_pct"] == pytest.approx(25.0)
        # ld_pct: 10 <= angle < 25 → 0/4
        assert result["ld_pct"] == pytest.approx(0.0)
        # sweet_spot_pct: 8 <= angle <= 32 → 2/4 (8, 28)
        assert result["sweet_spot_pct"] == pytest.approx(50.0)

    def test_exit_velo_p90_single_row(self) -> None:
        rows = [{"launch_speed": 100.0, "launch_angle": 20.0, "barrel": 0}]
        result = batted_ball_profile(rows)
        assert result["exit_velo_p90"] == pytest.approx(100.0)

    def test_angle_based_pcts_exclude_null_angle(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 5.0, "barrel": 0},
            {"launch_speed": 95.0, "launch_angle": None, "barrel": 0},
        ]
        result = batted_ball_profile(rows)
        # Only 1 row with angle; angle=5 → gb
        assert result["gb_pct"] == pytest.approx(100.0)
        assert result["fb_pct"] == pytest.approx(0.0)
        assert result["ld_pct"] == pytest.approx(0.0)
        # exit_velo_p90 uses all batted balls: sorted [95, 100] → p90 = 99.5
        assert result["exit_velo_p90"] == pytest.approx(99.5)

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
        assert len(result) == 10

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
            "gb_pct",
            "fb_pct",
            "ld_pct",
            "sweet_spot_pct",
            "exit_velo_p90",
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

    def test_null_launch_angle_with_valid_speed(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 1},
            {"launch_speed": 95.0, "launch_angle": None, "barrel": 0},
        ]
        result = batted_ball_profile(rows)
        # Both count for exit velo (n=2)
        assert result["avg_exit_velo"] == pytest.approx(97.5)
        # Only one has launch_angle
        assert result["avg_launch_angle"] == pytest.approx(25.0)


class TestBattedBallTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(BATTED_BALL, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert BATTED_BALL.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(BATTED_BALL.outputs) == 10

    def test_transform_callable(self) -> None:
        assert BATTED_BALL.transform is batted_ball_profile
