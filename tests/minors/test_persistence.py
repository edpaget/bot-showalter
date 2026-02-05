"""Tests for minors/persistence.py - MLE model storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fantasy_baseball_manager.minors.model import (
    MLEGradientBoostingModel,
    MLEStatModel,
)
from fantasy_baseball_manager.minors.persistence import (
    MLEModelMetadata,
    MLEModelStore,
)


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory."""
    model_dir = tmp_path / "models" / "mle"
    model_dir.mkdir(parents=True)
    return model_dir


@pytest.fixture
def trained_model_set() -> MLEGradientBoostingModel:
    """Create a trained model set for testing."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    feature_names = ["f1", "f2", "f3"]

    model_set = MLEGradientBoostingModel(
        player_type="batter",
        feature_names=feature_names,
        training_years=(2022, 2023),
    )

    for stat in ["hr", "so", "bb"]:
        y = np.random.randn(50) * 0.01
        model = MLEStatModel(stat_name=stat)
        model.fit(X, y, feature_names)
        model_set.add_model(model)

    return model_set


class TestMLEModelStore:
    """Tests for MLEModelStore."""

    def test_save_and_load(
        self,
        temp_model_dir: Path,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should save and load models correctly."""
        store = MLEModelStore(model_dir=temp_model_dir)

        # Save
        model_path = store.save(trained_model_set, name="test")

        assert model_path.exists()
        assert (temp_model_dir / "test_batter_meta.json").exists()

        # Load
        loaded = store.load("test", player_type="batter")

        assert loaded.player_type == "batter"
        assert loaded.feature_names == trained_model_set.feature_names
        assert loaded.training_years == trained_model_set.training_years
        assert set(loaded.get_stats()) == set(trained_model_set.get_stats())

    def test_load_nonexistent_raises(self, temp_model_dir: Path) -> None:
        """Should raise FileNotFoundError for nonexistent model."""
        store = MLEModelStore(model_dir=temp_model_dir)

        with pytest.raises(FileNotFoundError, match="MLE model not found"):
            store.load("nonexistent")

    def test_exists(
        self,
        temp_model_dir: Path,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should correctly check if model exists."""
        store = MLEModelStore(model_dir=temp_model_dir)

        assert not store.exists("test")

        store.save(trained_model_set, name="test")

        assert store.exists("test")
        assert not store.exists("other")

    def test_get_metadata(
        self,
        temp_model_dir: Path,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should retrieve metadata without loading full model."""
        store = MLEModelStore(model_dir=temp_model_dir)
        store.save(trained_model_set, name="test")

        metadata = store.get_metadata("test")

        assert metadata is not None
        assert metadata.name == "test"
        assert metadata.player_type == "batter"
        assert metadata.training_years == (2022, 2023)
        assert set(metadata.stats) == {"hr", "so", "bb"}
        assert metadata.feature_names == ["f1", "f2", "f3"]
        assert metadata.created_at is not None

    def test_get_metadata_nonexistent(self, temp_model_dir: Path) -> None:
        """Should return None for nonexistent model metadata."""
        store = MLEModelStore(model_dir=temp_model_dir)

        assert store.get_metadata("nonexistent") is None

    def test_list_models(
        self,
        temp_model_dir: Path,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should list all saved models."""
        store = MLEModelStore(model_dir=temp_model_dir)

        # Initially empty
        assert len(store.list_models()) == 0

        # Save two models
        store.save(trained_model_set, name="model1")

        # Create another model with different name
        trained_model_set2 = MLEGradientBoostingModel(
            player_type="batter",
            feature_names=["f1", "f2", "f3"],
            training_years=(2021,),
        )
        trained_model_set2.models = trained_model_set.models
        store.save(trained_model_set2, name="model2")

        models = store.list_models()

        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"model1", "model2"}

    def test_delete(
        self,
        temp_model_dir: Path,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should delete model and metadata."""
        store = MLEModelStore(model_dir=temp_model_dir)
        store.save(trained_model_set, name="test")

        assert store.exists("test")

        deleted = store.delete("test")

        assert deleted
        assert not store.exists("test")
        assert store.get_metadata("test") is None

    def test_delete_nonexistent(self, temp_model_dir: Path) -> None:
        """Should return False for nonexistent model."""
        store = MLEModelStore(model_dir=temp_model_dir)

        deleted = store.delete("nonexistent")

        assert not deleted

    def test_save_with_validation_metrics(
        self,
        temp_model_dir: Path,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should save validation metrics in metadata."""
        store = MLEModelStore(model_dir=temp_model_dir)

        validation_metrics = {
            "hr_rmse": 0.005,
            "so_rmse": 0.010,
            "bb_rmse": 0.008,
        }

        store.save(
            trained_model_set,
            name="test",
            validation_metrics=validation_metrics,
        )

        metadata = store.get_metadata("test")

        assert metadata is not None
        assert metadata.validation_metrics == validation_metrics

    def test_model_predictions_preserved_after_roundtrip(
        self,
        temp_model_dir: Path,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Predictions should be identical after save/load."""
        store = MLEModelStore(model_dir=temp_model_dir)

        np.random.seed(42)
        X_test = np.random.randn(10, 3)

        # Get predictions before save
        original_predictions = trained_model_set.predict_rates_batch(X_test)

        # Save and load
        store.save(trained_model_set, name="test")
        loaded = store.load("test")

        # Get predictions after load
        loaded_predictions = loaded.predict_rates_batch(X_test)

        for stat in original_predictions:
            np.testing.assert_array_almost_equal(
                original_predictions[stat],
                loaded_predictions[stat],
            )

    def test_directory_created_if_not_exists(self, tmp_path: Path) -> None:
        """Should create model directory if it doesn't exist."""
        model_dir = tmp_path / "new" / "nested" / "dir"
        assert not model_dir.exists()

        MLEModelStore(model_dir=model_dir)

        assert model_dir.exists()


class TestMLEModelMetadata:
    """Tests for MLEModelMetadata dataclass."""

    def test_frozen(self) -> None:
        """Metadata should be frozen (immutable)."""
        metadata = MLEModelMetadata(
            name="test",
            player_type="batter",
            training_years=(2022,),
            stats=["hr"],
            feature_names=["f1"],
            created_at="2024-01-01T00:00:00",
        )

        with pytest.raises(AttributeError):
            metadata.name = "other"  # type: ignore[misc]

    def test_optional_validation_metrics(self) -> None:
        """Validation metrics should be optional."""
        metadata = MLEModelMetadata(
            name="test",
            player_type="batter",
            training_years=(2022,),
            stats=["hr"],
            feature_names=["f1"],
            created_at="2024-01-01T00:00:00",
        )

        assert metadata.validation_metrics is None

    def test_with_validation_metrics(self) -> None:
        """Should store validation metrics."""
        metrics = {"hr_rmse": 0.005}
        metadata = MLEModelMetadata(
            name="test",
            player_type="batter",
            training_years=(2022,),
            stats=["hr"],
            feature_names=["f1"],
            created_at="2024-01-01T00:00:00",
            validation_metrics=metrics,
        )

        assert metadata.validation_metrics == metrics
