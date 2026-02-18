from __future__ import annotations

import math
from typing import Any

from fantasy_baseball_manager.features.transforms.batted_ball_interactions import (
    BATTED_BALL_INTERACTIONS,
    batted_ball_interaction_profile,
)
from fantasy_baseball_manager.features.types import DerivedTransformFeature


class TestBattedBallInteractionProfile:
    def test_computes_all_three_interactions(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "barrel_pct": 0.10,
                "avg_launch_angle": 12.0,
                "hard_hit_pct": 0.40,
                "pull_pct": 0.45,
                "avg_exit_velo": 90.0,
                "zone_contact_pct": 0.85,
            }
        ]
        result = batted_ball_interaction_profile(rows)
        assert result["barrel_la_interaction"] == 0.10 * 12.0
        assert result["hard_pull_interaction"] == 0.40 * 0.45
        assert result["quality_contact_interaction"] == 90.0 * 0.85

    def test_handles_nan_inputs(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "barrel_pct": float("nan"),
                "avg_launch_angle": 12.0,
                "hard_hit_pct": 0.40,
                "pull_pct": float("nan"),
                "avg_exit_velo": 90.0,
                "zone_contact_pct": float("nan"),
            }
        ]
        result = batted_ball_interaction_profile(rows)
        assert math.isnan(result["barrel_la_interaction"])
        assert math.isnan(result["hard_pull_interaction"])
        assert math.isnan(result["quality_contact_interaction"])

    def test_handles_zero_inputs(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "barrel_pct": 0.0,
                "avg_launch_angle": 0.0,
                "hard_hit_pct": 0.0,
                "pull_pct": 0.0,
                "avg_exit_velo": 0.0,
                "zone_contact_pct": 0.0,
            }
        ]
        result = batted_ball_interaction_profile(rows)
        assert result["barrel_la_interaction"] == 0.0
        assert result["hard_pull_interaction"] == 0.0
        assert result["quality_contact_interaction"] == 0.0

    def test_single_row_list(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "barrel_pct": 0.15,
                "avg_launch_angle": 14.0,
                "hard_hit_pct": 0.50,
                "pull_pct": 0.40,
                "avg_exit_velo": 92.0,
                "zone_contact_pct": 0.80,
            }
        ]
        result = batted_ball_interaction_profile(rows)
        assert result["barrel_la_interaction"] == 0.15 * 14.0
        assert result["hard_pull_interaction"] == 0.50 * 0.40
        assert result["quality_contact_interaction"] == 92.0 * 0.80

    def test_handles_none_inputs(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                "barrel_pct": None,
                "avg_launch_angle": 12.0,
                "hard_hit_pct": 0.40,
                "pull_pct": None,
                "avg_exit_velo": 90.0,
                "zone_contact_pct": None,
            }
        ]
        result = batted_ball_interaction_profile(rows)
        assert math.isnan(result["barrel_la_interaction"])
        assert math.isnan(result["hard_pull_interaction"])
        assert math.isnan(result["quality_contact_interaction"])


class TestBattedBallInteractionsFeature:
    def test_is_derived_transform_feature(self) -> None:
        assert isinstance(BATTED_BALL_INTERACTIONS, DerivedTransformFeature)

    def test_outputs_count(self) -> None:
        assert len(BATTED_BALL_INTERACTIONS.outputs) == 3

    def test_output_names(self) -> None:
        assert BATTED_BALL_INTERACTIONS.outputs == (
            "barrel_la_interaction",
            "hard_pull_interaction",
            "quality_contact_interaction",
        )

    def test_inputs(self) -> None:
        assert BATTED_BALL_INTERACTIONS.inputs == (
            "barrel_pct",
            "avg_launch_angle",
            "hard_hit_pct",
            "pull_pct",
            "avg_exit_velo",
            "zone_contact_pct",
        )

    def test_transform_callable(self) -> None:
        assert BATTED_BALL_INTERACTIONS.transform is batted_ball_interaction_profile
