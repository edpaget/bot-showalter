"""Tests for MLE model persistence via BaseModelStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from fantasy_baseball_manager.minors.model import (
    MLEGradientBoostingModel,
    MLEStatModel,
)
from fantasy_baseball_manager.registry.base_store import BaseModelStore, ModelMetadata
from fantasy_baseball_manager.registry.serializers import JoblibSerializer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory."""
    model_dir = tmp_path / "models" / "mle"
    model_dir.mkdir(parents=True)
    return model_dir


@pytest.fixture
def store(temp_model_dir: Path) -> BaseModelStore:
    """Create a BaseModelStore configured for MLE models."""
    return BaseModelStore(
        model_dir=temp_model_dir,
        serializer=JoblibSerializer(),
        model_type_name="mle",
    )


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


class TestMLEPersistence:
    """Tests for MLE model persistence via BaseModelStore."""

    def test_save_and_load(
        self,
        store: BaseModelStore,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should save and load models correctly."""
        model_path = store.save_params(
            trained_model_set.get_params(),
            "test",
            trained_model_set.player_type,
            training_years=trained_model_set.training_years,
            stats=trained_model_set.get_stats(),
            feature_names=trained_model_set.feature_names,
        )

        assert model_path.exists()
        assert (store.model_dir / "test_batter_meta.json").exists()

        # Load
        loaded = MLEGradientBoostingModel.from_params(
            store.load_params("test", "batter")
        )

        assert loaded.player_type == "batter"
        assert loaded.feature_names == trained_model_set.feature_names
        assert loaded.training_years == trained_model_set.training_years
        assert set(loaded.get_stats()) == set(trained_model_set.get_stats())

    def test_load_nonexistent_raises(self, store: BaseModelStore) -> None:
        """Should raise FileNotFoundError for nonexistent model."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            store.load_params("nonexistent", "batter")

    def test_exists(
        self,
        store: BaseModelStore,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should correctly check if model exists."""
        assert not store.exists("test", "batter")

        store.save_params(
            trained_model_set.get_params(),
            "test",
            trained_model_set.player_type,
            training_years=trained_model_set.training_years,
            stats=trained_model_set.get_stats(),
            feature_names=trained_model_set.feature_names,
        )

        assert store.exists("test", "batter")
        assert not store.exists("other", "batter")

    def test_get_metadata(
        self,
        store: BaseModelStore,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should retrieve metadata without loading full model."""
        store.save_params(
            trained_model_set.get_params(),
            "test",
            trained_model_set.player_type,
            training_years=trained_model_set.training_years,
            stats=trained_model_set.get_stats(),
            feature_names=trained_model_set.feature_names,
        )

        metadata = store.get_metadata("test", "batter")

        assert metadata is not None
        assert isinstance(metadata, ModelMetadata)
        assert metadata.name == "test"
        assert metadata.player_type == "batter"
        assert metadata.model_type == "mle"
        assert metadata.training_years == (2022, 2023)
        assert set(metadata.stats) == {"hr", "so", "bb"}
        assert metadata.feature_names == ["f1", "f2", "f3"]
        assert metadata.created_at is not None

    def test_get_metadata_nonexistent(self, store: BaseModelStore) -> None:
        """Should return None for nonexistent model metadata."""
        assert store.get_metadata("nonexistent", "batter") is None

    def test_list_models(
        self,
        store: BaseModelStore,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should list all saved models."""
        # Initially empty
        assert len(store.list_models()) == 0

        # Save two models
        store.save_params(
            trained_model_set.get_params(),
            "model1",
            trained_model_set.player_type,
            training_years=trained_model_set.training_years,
            stats=trained_model_set.get_stats(),
            feature_names=trained_model_set.feature_names,
        )

        # Create another model with different name
        trained_model_set2 = MLEGradientBoostingModel(
            player_type="batter",
            feature_names=["f1", "f2", "f3"],
            training_years=(2021,),
        )
        trained_model_set2.models = trained_model_set.models
        store.save_params(
            trained_model_set2.get_params(),
            "model2",
            trained_model_set2.player_type,
            training_years=trained_model_set2.training_years,
            stats=trained_model_set2.get_stats(),
            feature_names=trained_model_set2.feature_names,
        )

        models = store.list_models()

        assert len(models) == 2
        names = {m.name for m in models}
        assert names == {"model1", "model2"}

    def test_delete(
        self,
        store: BaseModelStore,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should delete model and metadata."""
        store.save_params(
            trained_model_set.get_params(),
            "test",
            trained_model_set.player_type,
            training_years=trained_model_set.training_years,
            stats=trained_model_set.get_stats(),
            feature_names=trained_model_set.feature_names,
        )

        assert store.exists("test", "batter")

        deleted = store.delete("test", "batter")

        assert deleted
        assert not store.exists("test", "batter")
        assert store.get_metadata("test", "batter") is None

    def test_delete_nonexistent(self, store: BaseModelStore) -> None:
        """Should return False for nonexistent model."""
        deleted = store.delete("nonexistent", "batter")

        assert not deleted

    def test_save_with_validation_metrics(
        self,
        store: BaseModelStore,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Should save validation metrics in metadata."""
        validation_metrics = {
            "hr_rmse": 0.005,
            "so_rmse": 0.010,
            "bb_rmse": 0.008,
        }

        store.save_params(
            trained_model_set.get_params(),
            "test",
            trained_model_set.player_type,
            training_years=trained_model_set.training_years,
            stats=trained_model_set.get_stats(),
            feature_names=trained_model_set.feature_names,
            metrics={"validation_metrics": validation_metrics},
        )

        metadata = store.get_metadata("test", "batter")

        assert metadata is not None
        assert metadata.metrics["validation_metrics"] == validation_metrics

    def test_model_predictions_preserved_after_roundtrip(
        self,
        store: BaseModelStore,
        trained_model_set: MLEGradientBoostingModel,
    ) -> None:
        """Predictions should be identical after save/load."""
        np.random.seed(42)
        X_test = np.random.randn(10, 3)

        # Get predictions before save
        original_predictions = trained_model_set.predict_rates_batch(X_test)

        # Save and load
        store.save_params(
            trained_model_set.get_params(),
            "test",
            trained_model_set.player_type,
            training_years=trained_model_set.training_years,
            stats=trained_model_set.get_stats(),
            feature_names=trained_model_set.feature_names,
        )
        loaded = MLEGradientBoostingModel.from_params(
            store.load_params("test", "batter")
        )

        # Get predictions after load
        loaded_predictions = loaded.predict_rates_batch(X_test)

        for stat in original_predictions:
            np.testing.assert_array_almost_equal(
                original_predictions[stat],
                loaded_predictions[stat],
            )
