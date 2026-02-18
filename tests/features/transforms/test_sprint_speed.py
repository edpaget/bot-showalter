import math

import pytest

from fantasy_baseball_manager.features.transforms.sprint_speed import (
    SPRINT_SPEED_TRANSFORM,
    sprint_speed_profile,
)
from fantasy_baseball_manager.features.types import Source, TransformFeature


class TestSprintSpeedProfile:
    def test_basic_value(self) -> None:
        rows = [{"sprint_speed": 28.5}]
        result = sprint_speed_profile(rows)
        assert result["sprint_speed"] == pytest.approx(28.5)

    def test_empty_rows(self) -> None:
        result = sprint_speed_profile([])
        assert math.isnan(result["sprint_speed"])

    def test_none_speed(self) -> None:
        rows = [{"sprint_speed": None}]
        result = sprint_speed_profile(rows)
        assert math.isnan(result["sprint_speed"])

    def test_nan_speed(self) -> None:
        rows = [{"sprint_speed": float("nan")}]
        result = sprint_speed_profile(rows)
        assert math.isnan(result["sprint_speed"])

    def test_uses_first_row(self) -> None:
        rows = [{"sprint_speed": 27.0}, {"sprint_speed": 29.0}]
        result = sprint_speed_profile(rows)
        assert result["sprint_speed"] == pytest.approx(27.0)

    def test_output_key(self) -> None:
        result = sprint_speed_profile([])
        assert set(result.keys()) == {"sprint_speed"}

    def test_missing_key(self) -> None:
        rows = [{"other": 123}]
        result = sprint_speed_profile(rows)
        assert math.isnan(result["sprint_speed"])


class TestSprintSpeedTransformFeature:
    def test_is_transform_feature(self) -> None:
        assert isinstance(SPRINT_SPEED_TRANSFORM, TransformFeature)

    def test_source_is_sprint_speed(self) -> None:
        assert SPRINT_SPEED_TRANSFORM.source == Source.SPRINT_SPEED

    def test_outputs(self) -> None:
        assert SPRINT_SPEED_TRANSFORM.outputs == ("sprint_speed",)

    def test_columns(self) -> None:
        assert SPRINT_SPEED_TRANSFORM.columns == ("sprint_speed",)

    def test_transform_callable(self) -> None:
        assert SPRINT_SPEED_TRANSFORM.transform is sprint_speed_profile

    def test_group_by(self) -> None:
        assert SPRINT_SPEED_TRANSFORM.group_by == ("player_id", "season")
