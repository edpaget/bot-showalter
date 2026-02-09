"""Backward compatibility tests for migrated model stores.

Verifies that the legacy ModelStore, MTLModelStore, and MLEModelStore
APIs produce identical results after migration to BaseModelStore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.minors.persistence import MLEModelMetadata, MLEModelStore
from fantasy_baseball_manager.ml.mtl.persistence import MTLModelMetadata, MTLModelStore
from fantasy_baseball_manager.ml.persistence import ModelMetadata, ModelStore

if TYPE_CHECKING:
    from pathlib import Path


class TestModelStoreBackwardCompat:
    """Legacy ModelStore (GB residual) API compatibility."""

    def test_save_creates_expected_files(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        # Simulate saving via the store's internal mechanism
        store._store.save_params(
            {"player_type": "batter", "models": {}, "feature_names": [], "training_years": []},
            "test",
            "batter",
            training_years=(2023,),
            stats=["hr"],
            feature_names=["barrel_rate"],
        )
        assert (tmp_path / "test_batter.joblib").exists()
        assert (tmp_path / "test_batter_meta.json").exists()

    def test_exists_delegates_correctly(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        assert store.exists("missing", "batter") is False
        store._store.save_params({"w": 1}, "default", "batter")
        assert store.exists("default", "batter") is True

    def test_get_metadata_returns_legacy_type(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        store._store.save_params(
            {"w": 1},
            "default",
            "batter",
            training_years=(2022, 2023),
            stats=["hr", "sb"],
            feature_names=["barrel_rate"],
            metrics={"validation": {"strategy": "holdout"}},
        )
        meta = store.get_metadata("default", "batter")
        assert meta is not None
        assert isinstance(meta, ModelMetadata)
        assert meta.name == "default"
        assert meta.player_type == "batter"
        assert meta.training_years == (2022, 2023)
        assert meta.stats == ["hr", "sb"]
        assert meta.feature_names == ["barrel_rate"]
        assert meta.validation == {"strategy": "holdout"}

    def test_get_metadata_none_for_missing(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        assert store.get_metadata("missing", "batter") is None

    def test_list_models_returns_legacy_type(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        store._store.save_params({"w": 1}, "default", "batter")
        store._store.save_params({"w": 2}, "default", "pitcher")

        models = store.list_models()
        assert len(models) == 2
        assert all(isinstance(m, ModelMetadata) for m in models)

    def test_delete_works(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        store._store.save_params({"w": 1}, "default", "batter")
        assert store.delete("default", "batter") is True
        assert store.exists("default", "batter") is False

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        assert store.delete("missing", "batter") is False

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        store = ModelStore(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Model not found"):
            store.load("missing", "batter")


class TestMTLModelStoreBackwardCompat:
    """Legacy MTLModelStore API compatibility."""

    def test_exists_delegates(self, tmp_path: Path) -> None:
        store = MTLModelStore(model_dir=tmp_path)
        assert store.exists("missing", "batter") is False
        store._store.save_params({"w": 1}, "default", "batter")
        assert store.exists("default", "batter") is True

    def test_get_metadata_returns_legacy_type(self, tmp_path: Path) -> None:
        store = MTLModelStore(model_dir=tmp_path)
        store._store.save_params(
            {"w": 1},
            "default",
            "batter",
            training_years=(2023,),
            stats=["hr"],
            feature_names=["xba"],
            metrics={"validation_metrics": {"hr_rmse": 0.05}},
        )
        meta = store.get_metadata("default", "batter")
        assert meta is not None
        assert isinstance(meta, MTLModelMetadata)
        assert meta.validation_metrics == {"hr_rmse": 0.05}

    def test_list_models_returns_legacy_type(self, tmp_path: Path) -> None:
        store = MTLModelStore(model_dir=tmp_path)
        store._store.save_params({"w": 1}, "default", "batter")
        models = store.list_models()
        assert len(models) == 1
        assert isinstance(models[0], MTLModelMetadata)

    def test_delete_works(self, tmp_path: Path) -> None:
        store = MTLModelStore(model_dir=tmp_path)
        store._store.save_params({"w": 1}, "default", "pitcher")
        assert store.delete("default", "pitcher") is True
        assert store.exists("default", "pitcher") is False


class TestMLEModelStoreBackwardCompat:
    """Legacy MLEModelStore API compatibility."""

    def test_exists_with_default_player_type(self, tmp_path: Path) -> None:
        store = MLEModelStore(model_dir=tmp_path)
        store._store.save_params({"w": 1}, "default", "batter")
        # Default player_type is "batter"
        assert store.exists("default") is True

    def test_get_metadata_returns_legacy_type(self, tmp_path: Path) -> None:
        store = MLEModelStore(model_dir=tmp_path)
        store._store.save_params(
            {"w": 1},
            "default",
            "batter",
            training_years=(2022,),
            stats=["hr"],
            feature_names=["age"],
            metrics={"validation_metrics": {"r2": 0.85}},
        )
        meta = store.get_metadata("default")
        assert meta is not None
        assert isinstance(meta, MLEModelMetadata)
        assert meta.validation_metrics == {"r2": 0.85}

    def test_list_models_returns_legacy_type(self, tmp_path: Path) -> None:
        store = MLEModelStore(model_dir=tmp_path)
        store._store.save_params({"w": 1}, "default", "batter")
        models = store.list_models()
        assert len(models) == 1
        assert isinstance(models[0], MLEModelMetadata)

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        store = MLEModelStore(model_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Model not found"):
            store.load("nonexistent")
