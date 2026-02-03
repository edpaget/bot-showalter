"""Tests for model persistence."""

from pathlib import Path

import numpy as np
import pytest

from fantasy_baseball_manager.ml.persistence import ModelStore
from fantasy_baseball_manager.ml.residual_model import ResidualModelSet, StatResidualModel


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model storage."""
    return tmp_path / "models"


@pytest.fixture
def trained_batter_model_set() -> ResidualModelSet:
    """Create a trained batter model set for testing."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    feature_names = ["f1", "f2", "f3"]

    model_set = ResidualModelSet(
        player_type="batter",
        feature_names=feature_names,
        training_years=(2020, 2021, 2022),
    )

    for stat in ["hr", "so"]:
        y = np.random.randn(50)
        model = StatResidualModel(stat_name=stat)
        model.fit(X, y, feature_names)
        model_set.add_model(model)

    return model_set


class TestModelStore:
    def test_save_and_load(
        self,
        temp_model_dir: Path,
        trained_batter_model_set: ResidualModelSet,
    ) -> None:
        store = ModelStore(model_dir=temp_model_dir)
        store.save(trained_batter_model_set, "test_model")

        loaded = store.load("test_model", "batter")

        assert loaded.player_type == "batter"
        assert loaded.training_years == (2020, 2021, 2022)
        assert loaded.feature_names == ["f1", "f2", "f3"]
        assert set(loaded.get_stats()) == {"hr", "so"}

    def test_exists(
        self,
        temp_model_dir: Path,
        trained_batter_model_set: ResidualModelSet,
    ) -> None:
        store = ModelStore(model_dir=temp_model_dir)

        assert not store.exists("test_model", "batter")

        store.save(trained_batter_model_set, "test_model")

        assert store.exists("test_model", "batter")
        assert not store.exists("test_model", "pitcher")
        assert not store.exists("other_model", "batter")

    def test_load_raises_on_missing(self, temp_model_dir: Path) -> None:
        store = ModelStore(model_dir=temp_model_dir)

        with pytest.raises(FileNotFoundError):
            store.load("nonexistent", "batter")

    def test_get_metadata(
        self,
        temp_model_dir: Path,
        trained_batter_model_set: ResidualModelSet,
    ) -> None:
        store = ModelStore(model_dir=temp_model_dir)
        store.save(trained_batter_model_set, "test_model")

        meta = store.get_metadata("test_model", "batter")

        assert meta is not None
        assert meta.name == "test_model"
        assert meta.player_type == "batter"
        assert meta.training_years == (2020, 2021, 2022)
        assert set(meta.stats) == {"hr", "so"}
        assert meta.feature_names == ["f1", "f2", "f3"]
        assert meta.created_at is not None

    def test_get_metadata_returns_none_on_missing(self, temp_model_dir: Path) -> None:
        store = ModelStore(model_dir=temp_model_dir)
        meta = store.get_metadata("nonexistent", "batter")
        assert meta is None

    def test_list_models(self, temp_model_dir: Path) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 3)
        feature_names = ["f1", "f2", "f3"]

        store = ModelStore(model_dir=temp_model_dir)

        # Save batter model
        batter_set = ResidualModelSet(
            player_type="batter",
            feature_names=feature_names,
            training_years=(2021,),
        )
        model = StatResidualModel(stat_name="hr")
        model.fit(X, np.random.randn(50), feature_names)
        batter_set.add_model(model)
        store.save(batter_set, "model_a")

        # Save pitcher model
        pitcher_set = ResidualModelSet(
            player_type="pitcher",
            feature_names=feature_names,
            training_years=(2022,),
        )
        model = StatResidualModel(stat_name="er")
        model.fit(X, np.random.randn(50), feature_names)
        pitcher_set.add_model(model)
        store.save(pitcher_set, "model_b")

        models = store.list_models()

        assert len(models) == 2
        names = {(m.name, m.player_type) for m in models}
        assert ("model_a", "batter") in names
        assert ("model_b", "pitcher") in names

    def test_delete(
        self,
        temp_model_dir: Path,
        trained_batter_model_set: ResidualModelSet,
    ) -> None:
        store = ModelStore(model_dir=temp_model_dir)
        store.save(trained_batter_model_set, "test_model")

        assert store.exists("test_model", "batter")

        result = store.delete("test_model", "batter")

        assert result is True
        assert not store.exists("test_model", "batter")

    def test_delete_returns_false_on_missing(self, temp_model_dir: Path) -> None:
        store = ModelStore(model_dir=temp_model_dir)
        result = store.delete("nonexistent", "batter")
        assert result is False

    def test_predictions_preserved_after_load(
        self,
        temp_model_dir: Path,
        trained_batter_model_set: ResidualModelSet,
    ) -> None:
        """Verify that loaded models produce the same predictions."""
        np.random.seed(42)
        X_test = np.random.randn(5, 3)

        # Get predictions before save
        original_predictions = trained_batter_model_set.predict_residuals(X_test[0])

        # Save and load
        store = ModelStore(model_dir=temp_model_dir)
        store.save(trained_batter_model_set, "test_model")
        loaded = store.load("test_model", "batter")

        # Get predictions after load
        loaded_predictions = loaded.predict_residuals(X_test[0])

        # Verify predictions match
        for stat in original_predictions:
            assert abs(original_predictions[stat] - loaded_predictions[stat]) < 1e-10
