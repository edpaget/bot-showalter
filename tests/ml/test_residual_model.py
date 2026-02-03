"""Tests for residual model classes."""

import numpy as np
import pytest

from fantasy_baseball_manager.ml.residual_model import (
    ModelHyperparameters,
    ResidualModelSet,
    StatResidualModel,
)


class TestStatResidualModel:
    def test_fit_and_predict(self) -> None:
        model = StatResidualModel(stat_name="hr")

        # Create simple training data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
        feature_names = ["f1", "f2", "f3", "f4", "f5"]

        model.fit(X, y, feature_names)

        assert model.is_fitted
        predictions = model.predict(X[:10])
        assert predictions.shape == (10,)

    def test_predict_raises_when_not_fitted(self) -> None:
        model = StatResidualModel(stat_name="hr")

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(np.array([[1, 2, 3, 4, 5]]))

    def test_feature_importances(self) -> None:
        model = StatResidualModel(stat_name="hr")

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 5 + np.random.randn(100) * 0.1  # f1 dominates
        feature_names = ["f1", "f2", "f3"]

        model.fit(X, y, feature_names)

        importances = model.feature_importances()
        assert set(importances.keys()) == {"f1", "f2", "f3"}
        # f1 should have highest importance
        assert importances["f1"] > importances["f2"]
        assert importances["f1"] > importances["f3"]

    def test_feature_importances_raises_when_not_fitted(self) -> None:
        model = StatResidualModel(stat_name="hr")

        with pytest.raises(ValueError, match="has not been fitted"):
            model.feature_importances()

    def test_custom_hyperparameters(self) -> None:
        hp = ModelHyperparameters(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
        )
        model = StatResidualModel(stat_name="hr", hyperparameters=hp)

        assert model.hyperparameters.n_estimators == 50
        assert model.hyperparameters.max_depth == 3
        assert model.hyperparameters.learning_rate == 0.05

    def test_get_params_and_from_params(self) -> None:
        model = StatResidualModel(stat_name="hr")

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        feature_names = ["f1", "f2", "f3"]

        model.fit(X, y, feature_names)
        original_predictions = model.predict(X[:5])

        # Serialize and deserialize
        params = model.get_params()
        restored = StatResidualModel.from_params(params)

        assert restored.stat_name == "hr"
        assert restored.is_fitted
        restored_predictions = restored.predict(X[:5])
        np.testing.assert_array_almost_equal(original_predictions, restored_predictions)


class TestResidualModelSet:
    def test_add_and_predict_residuals(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = ResidualModelSet(
            player_type="batter",
            feature_names=feature_names,
            training_years=(2020, 2021, 2022),
        )

        # Train and add models for two stats
        for stat in ["hr", "so"]:
            y = np.random.randn(100)
            model = StatResidualModel(stat_name=stat)
            model.fit(X, y, feature_names)
            model_set.add_model(model)

        # Predict residuals for a single feature vector
        test_features = X[0]
        residuals = model_set.predict_residuals(test_features)

        assert "hr" in residuals
        assert "so" in residuals
        assert isinstance(residuals["hr"], float)
        assert isinstance(residuals["so"], float)

    def test_get_stats(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = ResidualModelSet(player_type="pitcher")

        # Add one fitted model
        y = np.random.randn(50)
        model = StatResidualModel(stat_name="er")
        model.fit(X, y, feature_names)
        model_set.add_model(model)

        # Add one unfitted model
        unfitted = StatResidualModel(stat_name="h")
        model_set.models["h"] = unfitted

        stats = model_set.get_stats()
        assert stats == ["er"]

    def test_predict_residuals_handles_2d_input(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = ResidualModelSet(player_type="batter", feature_names=feature_names)

        model = StatResidualModel(stat_name="hr")
        model.fit(X, np.random.randn(50), feature_names)
        model_set.add_model(model)

        # Test with 2D input (1, n_features)
        test_features = X[0].reshape(1, -1)
        residuals = model_set.predict_residuals(test_features)

        assert "hr" in residuals

    def test_get_params_and_from_params(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = ResidualModelSet(
            player_type="batter",
            feature_names=feature_names,
            training_years=(2020, 2021),
        )

        model = StatResidualModel(stat_name="hr")
        model.fit(X, np.random.randn(50), feature_names)
        model_set.add_model(model)

        # Serialize and deserialize
        params = model_set.get_params()
        restored = ResidualModelSet.from_params(params)

        assert restored.player_type == "batter"
        assert restored.feature_names == feature_names
        assert restored.training_years == (2020, 2021)
        assert "hr" in restored.models
        assert restored.models["hr"].is_fitted
