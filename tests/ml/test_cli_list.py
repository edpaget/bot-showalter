"""Tests for `ml list` command with cross-type listing."""

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


def _invoke_list(*args: str, registry: ModelRegistry) -> Result:
    with patch("fantasy_baseball_manager.ml.cli._get_registry", return_value=registry):
        return runner.invoke(ml_app, ["list", *args])


class TestListEmpty:
    def test_list_empty(self, registry: ModelRegistry) -> None:
        result = _invoke_list(registry=registry)
        assert result.exit_code == 0
        assert "No trained models found." in result.output


class TestListShowsGBModel:
    def test_list_shows_gb_model(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "default",
            "batter",
            training_years=(2022, 2023),
            stats=["HR", "AVG"],
            version=1,
        )
        result = _invoke_list(registry=registry)
        assert result.exit_code == 0
        assert "default" in result.output
        assert "gb_residual" in result.output


class TestListShowsModelsAcrossTypes:
    def test_list_shows_models_across_types(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "default",
            "batter",
            training_years=(2023,),
            version=1,
        )
        registry.mtl_store.save_params(
            {"dummy": True},
            "experiment",
            "pitcher",
            training_years=(2022, 2023),
            version=1,
        )
        result = _invoke_list(registry=registry)
        assert result.exit_code == 0
        assert "default" in result.output
        assert "experiment" in result.output
        assert "gb_residual" in result.output
        assert "mtl" in result.output


class TestListShowsVersionColumn:
    def test_list_shows_version_column(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "default_v2",
            "batter",
            training_years=(2023,),
            version=2,
        )
        result = _invoke_list(registry=registry)
        assert result.exit_code == 0
        assert "Version" in result.output
        assert "2" in result.output


class TestListShowsModelTypeColumn:
    def test_list_shows_model_type_column(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "default",
            "batter",
            training_years=(2023,),
            version=1,
        )
        registry.mtl_store.save_params(
            {"dummy": True},
            "default",
            "pitcher",
            training_years=(2023,),
            version=1,
        )
        result = _invoke_list(registry=registry)
        assert result.exit_code == 0
        assert "Model Type" in result.output
        assert "gb_residual" in result.output
        assert "mtl" in result.output


class TestListFilterByType:
    def test_list_filter_by_type(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "gb_model",
            "batter",
            training_years=(2023,),
            version=1,
        )
        registry.mtl_store.save_params(
            {"dummy": True},
            "mtl_model",
            "pitcher",
            training_years=(2023,),
            version=1,
        )
        result = _invoke_list("--type", "mtl", registry=registry)
        assert result.exit_code == 0
        assert "mtl_model" in result.output
        assert "gb_model" not in result.output


class TestListFilterUnknownType:
    def test_list_filter_unknown_type_shows_empty(self, registry: ModelRegistry) -> None:
        registry.gb_store.save_params(
            {"dummy": True},
            "default",
            "batter",
            training_years=(2023,),
            version=1,
        )
        result = _invoke_list("--type", "bogus", registry=registry)
        assert result.exit_code == 0
        assert "No trained models found." in result.output
