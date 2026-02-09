"""Tests for ModelRegistry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.registry.base_store import BaseModelStore
from fantasy_baseball_manager.registry.mtl_store import MTLBaseModelStore
from fantasy_baseball_manager.registry.registry import ModelRegistry
from fantasy_baseball_manager.registry.serializers import JoblibSerializer

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _FakeContextualMetadata:
    """Minimal stand-in for ContextualModelMetadata."""

    name: str
    epoch: int = 10
    train_loss: float = 0.1
    val_loss: float = 0.2
    pitch_type_accuracy: float | None = None
    pitch_result_accuracy: float | None = None
    created_at: str | None = "2024-01-01T00:00:00"
    perspective: str | None = None
    target_stats: tuple[str, ...] | None = None
    per_stat_mse: dict[str, float] | None = None
    base_model: str | None = None


@dataclass
class _FakeContextualStore:
    """Minimal stand-in for ContextualModelStore."""

    checkpoints: list[_FakeContextualMetadata] = field(default_factory=list)

    def list_checkpoints(self) -> list[_FakeContextualMetadata]:
        return self.checkpoints


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    serializer = JoblibSerializer()
    gb = BaseModelStore(model_dir=tmp_path / "gb", serializer=serializer, model_type_name="gb_residual")
    mtl = MTLBaseModelStore(model_dir=tmp_path / "mtl", serializer=serializer, model_type_name="mtl")
    mle = BaseModelStore(model_dir=tmp_path / "mle", serializer=serializer, model_type_name="mle")
    ctx = _FakeContextualStore()
    return ModelRegistry(gb_store=gb, mtl_store=mtl, mle_store=mle, contextual_store=ctx)  # type: ignore[arg-type]


class TestListAll:
    def test_empty_registry(self, registry: ModelRegistry) -> None:
        assert registry.list_all() == []

    def test_lists_across_stores(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "default", "batter")
        registry.mle_store.save_params(params, "default", "batter")

        models = registry.list_all()
        assert len(models) == 2
        types = {m.model_type for m in models}
        assert types == {"gb_residual", "mle"}

    def test_filter_by_model_type(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "default", "batter")
        registry.mle_store.save_params(params, "default", "batter")

        models = registry.list_all(model_type="gb_residual")
        assert len(models) == 1
        assert models[0].model_type == "gb_residual"

    def test_filter_by_player_type(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "default", "batter")
        registry.gb_store.save_params(params, "default", "pitcher")

        models = registry.list_all(player_type="batter")
        assert len(models) == 1
        assert models[0].player_type == "batter"

    def test_includes_contextual(self, registry: ModelRegistry) -> None:
        registry.contextual_store.checkpoints.append(  # type: ignore[attr-defined]
            _FakeContextualMetadata(name="pretrain_best", perspective="batter")
        )
        models = registry.list_all(model_type="contextual")
        assert len(models) == 1
        assert models[0].model_type == "contextual"
        assert models[0].player_type == "batter"


class TestVersioning:
    def test_next_version_no_existing(self, registry: ModelRegistry) -> None:
        assert registry.next_version("default", "gb_residual", "batter") == 1

    def test_next_version_increments(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "default", "batter", version=1)
        registry.gb_store.save_params(params, "default_v2", "batter", version=2)

        assert registry.next_version("default", "gb_residual", "batter") == 3

    def test_versions_of_returns_sorted(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "default_v2", "batter", version=2)
        registry.gb_store.save_params(params, "default", "batter", version=1)
        registry.gb_store.save_params(params, "default_v3", "batter", version=3)

        versions = registry.versions_of("default", "gb_residual", "batter")
        assert [v.version for v in versions] == [1, 2, 3]

    def test_versions_of_filters_player_type(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "default", "batter", version=1)
        registry.gb_store.save_params(params, "default", "pitcher", version=1)

        versions = registry.versions_of("default", "gb_residual", "pitcher")
        assert len(versions) == 1
        assert versions[0].player_type == "pitcher"


class TestCompare:
    def test_compare_models(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "v1", "batter", metrics={"rmse": 0.10})
        registry.gb_store.save_params(params, "v2", "batter", metrics={"rmse": 0.08})

        result = registry.compare("v1", "v2", "gb_residual", "batter")
        assert result["a"]["name"] == "v1"
        assert result["b"]["name"] == "v2"
        assert result["metrics_diff"]["rmse"]["delta"] == pytest.approx(-0.02)

    def test_compare_missing_model_raises(self, registry: ModelRegistry) -> None:
        params = {"w": 1}
        registry.gb_store.save_params(params, "v1", "batter")

        with pytest.raises(FileNotFoundError):
            registry.compare("v1", "missing", "gb_residual", "batter")


class TestGetStore:
    def test_valid_types(self, registry: ModelRegistry) -> None:
        assert registry.get_store("gb_residual") is registry.gb_store
        assert registry.get_store("mtl") is registry.mtl_store
        assert registry.get_store("mle") is registry.mle_store

    def test_invalid_type_raises(self, registry: ModelRegistry) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            registry.get_store("contextual")

    def test_unknown_type_raises(self, registry: ModelRegistry) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            registry.get_store("nonexistent")
