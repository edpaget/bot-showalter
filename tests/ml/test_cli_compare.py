"""Tests for `ml compare` command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.ml.cli import ml_app
from fantasy_baseball_manager.registry.base_store import BaseModelStore
from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore
from fantasy_baseball_manager.registry.registry import ModelRegistry
from fantasy_baseball_manager.registry.serializers import JoblibSerializer

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import Result

runner = CliRunner()


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    from dataclasses import dataclass, field

    @dataclass
    class _FakeContextualStore:
        checkpoints: list = field(default_factory=list)

        def list_checkpoints(self) -> list:
            return self.checkpoints

    serializer = JoblibSerializer()
    gb = BaseModelStore(model_dir=tmp_path / "gb", serializer=serializer, model_type_name="gb_residual")
    mtl = MTLBaseModelStore(model_dir=tmp_path / "mtl", serializer=serializer, model_type_name="mtl")
    mle = BaseModelStore(model_dir=tmp_path / "mle", serializer=serializer, model_type_name="mle")
    ctx = _FakeContextualStore()
    return ModelRegistry(gb_store=gb, mtl_store=mtl, mle_store=mle, contextual_store=ctx)  # type: ignore[arg-type]


def _invoke_compare(*args: str, registry: ModelRegistry) -> Result:
    with patch("fantasy_baseball_manager.ml.cli._get_registry", return_value=registry):
        return runner.invoke(ml_app, ["compare", *args])


class TestCompareTwoModels:
    def test_compare_two_models(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "model_a",
            "batter",
            training_years=(2021, 2022, 2023),
            stats=["HR", "AVG"],
            metrics={"rmse": 0.10},
            version=1,
        )
        registry.gb_store.save_params(
            {"dummy": True},
            "model_b",
            "batter",
            training_years=(2021, 2022, 2023),
            stats=["HR", "AVG"],
            metrics={"rmse": 0.08},
            version=2,
        )
        result = _invoke_compare("model_a", "model_b", registry=registry)
        assert result.exit_code == 0
        assert "model_a" in result.output
        assert "model_b" in result.output
        # Delta should be shown
        assert "Delta" in result.output


class TestCompareMissingModel:
    def test_compare_missing_model(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "model_a",
            "batter",
            training_years=(2021, 2022, 2023),
            stats=["HR"],
            version=1,
        )
        result = _invoke_compare("model_a", "nonexistent", registry=registry)
        assert result.exit_code == 1
        assert "Error" in result.output


class TestCompareDifferentModelType:
    def test_compare_different_model_type(self, registry: ModelRegistry) -> None:
        registry.mtl_store.save_params(
            {"dummy": True},
            "mtl_a",
            "batter",
            training_years=(2022, 2023),
            stats=["HR"],
            version=1,
        )
        registry.mtl_store.save_params(
            {"dummy": True},
            "mtl_b",
            "batter",
            training_years=(2022, 2023),
            stats=["HR"],
            version=2,
        )
        result = _invoke_compare("mtl_a", "mtl_b", "--model-type", "mtl", registry=registry)
        assert result.exit_code == 0
        assert "mtl_a" in result.output
        assert "mtl_b" in result.output


class TestCompareShowsTrainingYearsDiff:
    def test_compare_shows_training_years_diff(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "old_model",
            "batter",
            training_years=(2021, 2022, 2023),
            stats=["HR"],
            version=1,
        )
        registry.gb_store.save_params(
            {"dummy": True},
            "new_model",
            "batter",
            training_years=(2022, 2023, 2024),
            stats=["HR"],
            version=2,
        )
        result = _invoke_compare("old_model", "new_model", registry=registry)
        assert result.exit_code == 0
        assert "2021" in result.output
        assert "2024" in result.output
