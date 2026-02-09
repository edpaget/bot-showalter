"""Tests for BaseModelStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.registry.base_store import BaseModelStore, ModelMetadata
from fantasy_baseball_manager.registry.serializers import JoblibSerializer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def store(tmp_path: Path) -> BaseModelStore:
    return BaseModelStore(
        model_dir=tmp_path / "models",
        serializer=JoblibSerializer(),
        model_type_name="test_model",
    )


@pytest.fixture
def sample_params() -> dict:
    return {"weights": [1.0, 2.0], "bias": 0.5}


class TestSaveAndLoad:
    def test_round_trip(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(
            sample_params,
            "default",
            "batter",
            training_years=(2022, 2023),
            stats=["hr", "sb"],
            feature_names=["barrel_rate"],
        )

        loaded = store.load_params("default", "batter")
        assert loaded == sample_params

    def test_save_creates_metadata_file(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(sample_params, "my_model", "pitcher")
        meta_path = store.model_dir / "my_model_pitcher_meta.json"
        assert meta_path.exists()

    def test_save_creates_model_file(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(sample_params, "my_model", "pitcher")
        model_path = store.model_dir / "my_model_pitcher.joblib"
        assert model_path.exists()

    def test_save_returns_model_path(self, store: BaseModelStore, sample_params: dict) -> None:
        path = store.save_params(sample_params, "default", "batter")
        assert path == store.model_dir / "default_batter.joblib"

    def test_load_nonexistent_raises(self, store: BaseModelStore) -> None:
        with pytest.raises(FileNotFoundError, match="Model not found"):
            store.load_params("missing", "batter")


class TestMetadata:
    def test_get_metadata(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(
            sample_params,
            "default",
            "batter",
            training_years=(2021, 2022, 2023),
            stats=["hr", "sb"],
            feature_names=["barrel_rate", "exit_velo"],
            metrics={"rmse": 0.05},
            version=2,
        )

        meta = store.get_metadata("default", "batter")
        assert meta is not None
        assert meta.name == "default"
        assert meta.model_type == "test_model"
        assert meta.player_type == "batter"
        assert meta.version == 2
        assert meta.training_years == (2021, 2022, 2023)
        assert meta.stats == ["hr", "sb"]
        assert meta.feature_names == ["barrel_rate", "exit_velo"]
        assert meta.metrics == {"rmse": 0.05}
        assert meta.created_at  # non-empty timestamp

    def test_get_metadata_nonexistent_returns_none(self, store: BaseModelStore) -> None:
        assert store.get_metadata("missing", "batter") is None

    def test_metadata_defaults(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(sample_params, "simple", "batter")
        meta = store.get_metadata("simple", "batter")
        assert meta is not None
        assert meta.version == 1
        assert meta.training_years == ()
        assert meta.stats == []
        assert meta.feature_names == []
        assert meta.metrics == {}


class TestExists:
    def test_exists_true(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(sample_params, "default", "batter")
        assert store.exists("default", "batter") is True

    def test_exists_false(self, store: BaseModelStore) -> None:
        assert store.exists("missing", "batter") is False

    def test_exists_wrong_player_type(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(sample_params, "default", "batter")
        assert store.exists("default", "pitcher") is False


class TestListModels:
    def test_empty_store(self, store: BaseModelStore) -> None:
        assert store.list_models() == []

    def test_list_multiple(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(sample_params, "default", "batter", stats=["hr"])
        store.save_params(sample_params, "default", "pitcher", stats=["so"])
        store.save_params(sample_params, "experiment", "batter", stats=["hr", "sb"])

        models = store.list_models()
        assert len(models) == 3
        names = {(m.name, m.player_type) for m in models}
        assert names == {("default", "batter"), ("default", "pitcher"), ("experiment", "batter")}


class TestDelete:
    def test_delete_existing(self, store: BaseModelStore, sample_params: dict) -> None:
        store.save_params(sample_params, "default", "batter")
        assert store.delete("default", "batter") is True
        assert store.exists("default", "batter") is False
        assert store.get_metadata("default", "batter") is None

    def test_delete_nonexistent(self, store: BaseModelStore) -> None:
        assert store.delete("missing", "batter") is False


class TestModelMetadata:
    def test_to_dict_roundtrip(self) -> None:
        meta = ModelMetadata(
            name="test",
            model_type="gb_residual",
            player_type="batter",
            version=3,
            training_years=(2020, 2021),
            stats=["hr"],
            feature_names=["barrel_rate"],
            created_at="2024-01-01T00:00:00",
            metrics={"rmse": 0.1},
        )
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)
        assert restored == meta

    def test_from_dict_legacy_format(self) -> None:
        """Legacy metadata files lack model_type and version."""
        legacy = {
            "name": "default",
            "player_type": "batter",
            "training_years": [2022, 2023],
            "stats": ["hr", "sb"],
            "feature_names": ["barrel_rate"],
            "created_at": "2024-01-01T00:00:00",
        }
        meta = ModelMetadata.from_dict(legacy)
        assert meta.model_type == "unknown"
        assert meta.version == 1
        assert meta.metrics == {}
