from __future__ import annotations

import math
from typing import Any

import pytest

from fantasy_baseball_manager.features.transforms.age_interactions import (
    AGE_INTERACTIONS,
    age_interaction_profile,
)
from fantasy_baseball_manager.features.types import DerivedTransformFeature


class TestAgeInteractionProfile:
    def test_young_player(self) -> None:
        rows: list[dict[str, Any]] = [{"age": 24, "avg_1": 0.300, "obp_1": 0.380, "slg_1": 0.500}]
        result = age_interaction_profile(rows)
        assert result["age_avg_interact"] == pytest.approx((24 - 29) * 0.300)
        assert result["age_obp_interact"] == pytest.approx((24 - 29) * 0.380)
        assert result["age_slg_interact"] == pytest.approx((24 - 29) * 0.500)
        # Young player with good stats → negative interactions
        assert result["age_avg_interact"] < 0
        assert result["age_obp_interact"] < 0
        assert result["age_slg_interact"] < 0

    def test_old_player(self) -> None:
        rows: list[dict[str, Any]] = [{"age": 35, "avg_1": 0.280, "obp_1": 0.350, "slg_1": 0.450}]
        result = age_interaction_profile(rows)
        assert result["age_avg_interact"] == pytest.approx((35 - 29) * 0.280)
        assert result["age_obp_interact"] == pytest.approx((35 - 29) * 0.350)
        assert result["age_slg_interact"] == pytest.approx((35 - 29) * 0.450)
        # Old player with good stats → positive interactions
        assert result["age_avg_interact"] > 0
        assert result["age_obp_interact"] > 0
        assert result["age_slg_interact"] > 0

    def test_peak_age_player(self) -> None:
        rows: list[dict[str, Any]] = [{"age": 29, "avg_1": 0.300, "obp_1": 0.380, "slg_1": 0.500}]
        result = age_interaction_profile(rows)
        assert result["age_avg_interact"] == pytest.approx(0.0)
        assert result["age_obp_interact"] == pytest.approx(0.0)
        assert result["age_slg_interact"] == pytest.approx(0.0)

    def test_null_age(self) -> None:
        rows: list[dict[str, Any]] = [{"age": None, "avg_1": 0.300, "obp_1": 0.380, "slg_1": 0.500}]
        result = age_interaction_profile(rows)
        assert math.isnan(result["age_avg_interact"])
        assert math.isnan(result["age_obp_interact"])
        assert math.isnan(result["age_slg_interact"])

    def test_null_stats(self) -> None:
        rows: list[dict[str, Any]] = [{"age": 28, "avg_1": None, "obp_1": None, "slg_1": None}]
        result = age_interaction_profile(rows)
        assert math.isnan(result["age_avg_interact"])
        assert math.isnan(result["age_obp_interact"])
        assert math.isnan(result["age_slg_interact"])

    def test_nan_input(self) -> None:
        rows: list[dict[str, Any]] = [{"age": 28, "avg_1": float("nan"), "obp_1": 0.380, "slg_1": 0.500}]
        result = age_interaction_profile(rows)
        assert math.isnan(result["age_avg_interact"])
        assert result["age_obp_interact"] == pytest.approx((28 - 29) * 0.380)
        assert result["age_slg_interact"] == pytest.approx((28 - 29) * 0.500)


class TestAgeInteractionsFeature:
    def test_is_derived_transform_feature(self) -> None:
        assert isinstance(AGE_INTERACTIONS, DerivedTransformFeature)

    def test_outputs(self) -> None:
        assert AGE_INTERACTIONS.outputs == ("age_avg_interact", "age_obp_interact", "age_slg_interact")

    def test_inputs(self) -> None:
        assert AGE_INTERACTIONS.inputs == ("age", "avg_1", "obp_1", "slg_1")

    def test_transform_callable(self) -> None:
        assert AGE_INTERACTIONS.transform is age_interaction_profile
