"""Tests for CLI version resolution logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.ml.cli import resolve_version
from fantasy_baseball_manager.registry.base_store import BaseModelStore
from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore
from fantasy_baseball_manager.registry.registry import ModelRegistry
from fantasy_baseball_manager.registry.serializers import JoblibSerializer

if TYPE_CHECKING:
    from pathlib import Path


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


class TestResolveVersion:
    def test_no_existing_models_returns_version_1(self, registry: ModelRegistry) -> None:
        version, versioned_name = resolve_version(registry, "default", "gb_residual", None)
        assert version == 1
        assert versioned_name == "default"

    def test_existing_v1_auto_increments_to_v2(self, registry: ModelRegistry) -> None:
        # Save a v1 model for batter
        registry.gb_store.save_params(
            {"dummy": True}, "default", "batter",
            training_years=(2023,), version=1,
        )
        version, versioned_name = resolve_version(registry, "default", "gb_residual", None)
        assert version == 2
        assert versioned_name == "default_v2"

    def test_explicit_version_2(self, registry: ModelRegistry) -> None:
        version, versioned_name = resolve_version(registry, "default", "gb_residual", 2)
        assert version == 2
        assert versioned_name == "default_v2"

    def test_explicit_version_1(self, registry: ModelRegistry) -> None:
        version, versioned_name = resolve_version(registry, "default", "gb_residual", 1)
        assert version == 1
        assert versioned_name == "default"

    def test_asymmetric_batter_pitcher_uses_max(self, registry: ModelRegistry) -> None:
        # Batter has v1 and v2, pitcher only has v1
        registry.gb_store.save_params(
            {"dummy": True}, "default", "batter",
            training_years=(2023,), version=1,
        )
        registry.gb_store.save_params(
            {"dummy": True}, "default_v2", "batter",
            training_years=(2023,), version=2,
        )
        registry.gb_store.save_params(
            {"dummy": True}, "default", "pitcher",
            training_years=(2023,), version=1,
        )
        # Batter next = 3, pitcher next = 2 → max = 3
        version, versioned_name = resolve_version(registry, "default", "gb_residual", None)
        assert version == 3
        assert versioned_name == "default_v3"

    def test_mtl_model_type(self, registry: ModelRegistry) -> None:
        registry.mtl_store.save_params(
            {"dummy": True}, "default", "batter",
            training_years=(2023,), version=1,
        )
        version, versioned_name = resolve_version(registry, "default", "mtl", None)
        assert version == 2
        assert versioned_name == "default_v2"
