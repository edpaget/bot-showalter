from __future__ import annotations

import math

import pytest

from fantasy_baseball_manager.features.transforms.batted_ball import (
    BATTED_BALL,
    BATTED_BALL_AGAINST,
    SPRAY_ANGLE,
    batted_ball_against_profile,
    batted_ball_profile,
    spray_angle_profile,
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


class TestBattedBallAgainstProfile:
    def test_basic_metrics(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 1},
            {"launch_speed": 90.0, "launch_angle": 5.0, "barrel": 0},
            {"launch_speed": 95.0, "launch_angle": 15.0, "barrel": 0},
            {"launch_speed": 105.0, "launch_angle": 30.0, "barrel": 1},
        ]
        result = batted_ball_against_profile(rows)
        # avg_exit_velo_against: mean(100, 90, 95, 105) = 97.5
        assert result["avg_exit_velo_against"] == pytest.approx(97.5)
        # barrel_pct_against: 2/4 = 50%
        assert result["barrel_pct_against"] == pytest.approx(50.0)
        # gb_pct_against: angle < 10 → 1/4 (5)
        assert result["gb_pct_against"] == pytest.approx(25.0)
        # fb_pct_against: 25 <= angle < 50 → 2/4 (25, 30)
        assert result["fb_pct_against"] == pytest.approx(50.0)

    def test_empty_rows(self) -> None:
        result = batted_ball_against_profile([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 4

    def test_no_batted_ball_events(self) -> None:
        rows = [
            {"launch_speed": None, "launch_angle": None, "barrel": None},
        ]
        result = batted_ball_against_profile(rows)
        assert all(math.isnan(v) for v in result.values())

    def test_filters_null_launch_speed(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 25.0, "barrel": 1},
            {"launch_speed": None, "launch_angle": None, "barrel": None},
        ]
        result = batted_ball_against_profile(rows)
        assert result["avg_exit_velo_against"] == pytest.approx(100.0)
        assert result["barrel_pct_against"] == pytest.approx(100.0)

    def test_null_angle_excluded_from_gb_fb(self) -> None:
        rows = [
            {"launch_speed": 100.0, "launch_angle": 5.0, "barrel": 0},
            {"launch_speed": 95.0, "launch_angle": None, "barrel": 0},
        ]
        result = batted_ball_against_profile(rows)
        # Only 1 row with angle; angle=5 → gb
        assert result["gb_pct_against"] == pytest.approx(100.0)
        assert result["fb_pct_against"] == pytest.approx(0.0)
        # avg_exit_velo uses all batted balls
        assert result["avg_exit_velo_against"] == pytest.approx(97.5)

    def test_output_keys(self) -> None:
        result = batted_ball_against_profile([])
        expected_keys = {
            "gb_pct_against",
            "fb_pct_against",
            "avg_exit_velo_against",
            "barrel_pct_against",
        }
        assert set(result.keys()) == expected_keys

    def test_ground_ball_pitcher(self) -> None:
        rows = [
            {"launch_speed": 88.0, "launch_angle": -3.0, "barrel": 0},
            {"launch_speed": 90.0, "launch_angle": 5.0, "barrel": 0},
            {"launch_speed": 92.0, "launch_angle": 8.0, "barrel": 0},
        ]
        result = batted_ball_against_profile(rows)
        # All 3 angles < 10 → 100% gb
        assert result["gb_pct_against"] == pytest.approx(100.0)
        assert result["fb_pct_against"] == pytest.approx(0.0)
        assert result["barrel_pct_against"] == pytest.approx(0.0)


class TestBattedBallAgainstTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(BATTED_BALL_AGAINST, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert BATTED_BALL_AGAINST.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(BATTED_BALL_AGAINST.outputs) == 4

    def test_transform_callable(self) -> None:
        assert BATTED_BALL_AGAINST.transform is batted_ball_against_profile

    def test_name(self) -> None:
        assert BATTED_BALL_AGAINST.name == "batted_ball_against"


class TestSprayAngleProfile:
    def test_basic_metrics(self) -> None:
        # R batter: pull = left field (angle < -15), oppo = right field (angle > 15)
        # L batter: pull = right field (angle > 15), oppo = left field (angle < -15)
        # hc_x=125.42, hc_y=198.27 is dead center (angle=0)
        # Far left (low hc_x) → negative angle → pull for R
        # Far right (high hc_x) → positive angle → pull for L
        rows = [
            {"hc_x": 80.0, "hc_y": 150.0, "stand": "R"},  # left field → pull for R
            {"hc_x": 170.0, "hc_y": 150.0, "stand": "R"},  # right field → oppo for R
            {"hc_x": 125.42, "hc_y": 150.0, "stand": "R"},  # center
            {"hc_x": 170.0, "hc_y": 150.0, "stand": "L"},  # right field → pull for L
        ]
        result = spray_angle_profile(rows)
        # 4 batted balls: R-pull, R-oppo, R-center, L-pull
        # pull: 2/4 = 50%, oppo: 1/4 = 25%, center: 1/4 = 25%
        assert result["pull_pct"] == pytest.approx(50.0)
        assert result["oppo_pct"] == pytest.approx(25.0)
        assert result["center_pct"] == pytest.approx(25.0)

    def test_empty_rows(self) -> None:
        result = spray_angle_profile([])
        assert all(math.isnan(v) for v in result.values())
        assert len(result) == 3

    def test_all_pull(self) -> None:
        # All R batters hitting to left field (low hc_x)
        rows = [
            {"hc_x": 60.0, "hc_y": 150.0, "stand": "R"},
            {"hc_x": 70.0, "hc_y": 150.0, "stand": "R"},
        ]
        result = spray_angle_profile(rows)
        assert result["pull_pct"] == pytest.approx(100.0)
        assert result["oppo_pct"] == pytest.approx(0.0)
        assert result["center_pct"] == pytest.approx(0.0)

    def test_null_coordinates_excluded(self) -> None:
        rows = [
            {"hc_x": None, "hc_y": 150.0, "stand": "R"},
            {"hc_x": 60.0, "hc_y": None, "stand": "R"},
            {"hc_x": 60.0, "hc_y": 150.0, "stand": "R"},  # only valid row → pull
        ]
        result = spray_angle_profile(rows)
        assert result["pull_pct"] == pytest.approx(100.0)

    def test_null_stand_excluded(self) -> None:
        rows = [
            {"hc_x": 60.0, "hc_y": 150.0, "stand": None},
            {"hc_x": 60.0, "hc_y": 150.0, "stand": "R"},  # only valid row → pull
        ]
        result = spray_angle_profile(rows)
        assert result["pull_pct"] == pytest.approx(100.0)

    def test_output_keys(self) -> None:
        result = spray_angle_profile([])
        assert set(result.keys()) == {"pull_pct", "oppo_pct", "center_pct"}


class TestSprayAngleTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(SPRAY_ANGLE, TransformFeature)

    def test_source_is_statcast(self) -> None:
        assert SPRAY_ANGLE.source == Source.STATCAST

    def test_outputs_count(self) -> None:
        assert len(SPRAY_ANGLE.outputs) == 3

    def test_columns(self) -> None:
        assert SPRAY_ANGLE.columns == ("hc_x", "hc_y", "stand")

    def test_name(self) -> None:
        assert SPRAY_ANGLE.name == "spray_angle"
