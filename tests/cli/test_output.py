from typing import Any

import pytest

from fantasy_baseball_manager.cli._output import print_error, print_features
from fantasy_baseball_manager.features.types import (
    DeltaFeature,
    Feature,
    Source,
    TransformFeature,
)


def _plain_feature() -> Feature:
    return Feature(name="hr_1", source=Source.BATTING, column="hr", lag=0)


def _lag_feature() -> Feature:
    return Feature(name="hr_prev", source=Source.BATTING, column="hr", lag=1)


def _system_feature() -> Feature:
    return Feature(name="proj_hr", source=Source.PROJECTION, column="hr", system="steamer")


def _computed_feature() -> Feature:
    return Feature(name="age", source=Source.PLAYER, column="", computed="age")


def _delta_feature() -> DeltaFeature:
    left = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
    right = Feature(name="hr_2", source=Source.BATTING, column="hr", lag=2)
    return DeltaFeature(name="hr_diff", left=left, right=right)


def _transform_feature() -> TransformFeature:
    def dummy(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    return TransformFeature(
        name="rate_calc",
        source=Source.STATCAST,
        columns=("ev", "la"),
        group_by=("player_id", "season"),
        transform=dummy,
        outputs=("barrel_pct", "hard_hit_pct"),
    )


class TestPrintError:
    def test_print_error_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_error("something went wrong")
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "something went wrong" in captured.err
        assert captured.out == ""


class TestPrintFeatures:
    def test_print_features_plain(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_plain_feature(),))
        captured = capsys.readouterr()
        assert "batting.hr" in captured.out
        assert "1 features" in captured.out

    def test_print_features_lag(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_lag_feature(),))
        captured = capsys.readouterr()
        assert "lag=1" in captured.out

    def test_print_features_system(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_system_feature(),))
        captured = capsys.readouterr()
        assert "system=steamer" in captured.out

    def test_print_features_computed(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_computed_feature(),))
        captured = capsys.readouterr()
        assert "computed=age" in captured.out

    def test_print_features_delta(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_delta_feature(),))
        captured = capsys.readouterr()
        assert "delta(hr_1 - hr_2)" in captured.out

    def test_print_features_transform(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_features("test_model", (_transform_feature(),))
        captured = capsys.readouterr()
        assert "transform" in captured.out
        assert "barrel_pct" in captured.out
        assert "hard_hit_pct" in captured.out

    def test_print_features_mixed(self, capsys: pytest.CaptureFixture[str]) -> None:
        features = (
            _plain_feature(),
            _lag_feature(),
            _system_feature(),
            _computed_feature(),
            _delta_feature(),
            _transform_feature(),
        )
        print_features("test_model", features)
        captured = capsys.readouterr()
        assert "6 features" in captured.out
        assert "batting.hr" in captured.out
        assert "lag=1" in captured.out
        assert "system=steamer" in captured.out
        assert "computed=age" in captured.out
        assert "delta(hr_1 - hr_2)" in captured.out
        assert "barrel_pct, hard_hit_pct" in captured.out
