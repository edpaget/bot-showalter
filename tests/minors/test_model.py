"""Tests for minors/model.py - MLE gradient boosting models."""

import numpy as np
import pytest

from fantasy_baseball_manager.minors.model import (
    MLEGradientBoostingModel,
    MLEHyperparameters,
    MLEStatModel,
)


class TestMLEStatModel:
    """Tests for MLEStatModel."""

    def test_fit_and_predict(self) -> None:
        """Model should fit and predict correctly."""
        model = MLEStatModel(stat_name="hr")

        # Create simple training data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 0.01 + 0.03 + np.random.randn(100) * 0.005
        feature_names = ["f1", "f2", "f3", "f4", "f5"]

        model.fit(X, y, feature_names)

        assert model.is_fitted
        predictions = model.predict(X[:10])
        assert predictions.shape == (10,)

    def test_fit_with_sample_weight(self) -> None:
        """Model should accept sample weights."""
        model = MLEStatModel(stat_name="so")

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100) * 0.05 + 0.20
        feature_names = ["f1", "f2", "f3"]
        weights = np.abs(np.random.randn(100)) + 1  # Positive weights

        model.fit(X, y, feature_names, sample_weight=weights)

        assert model.is_fitted
        predictions = model.predict(X[:5])
        assert predictions.shape == (5,)

    def test_predict_raises_when_not_fitted(self) -> None:
        """Predict should raise when model not fitted."""
        model = MLEStatModel(stat_name="hr")

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(np.array([[1, 2, 3, 4, 5]]))

    def test_feature_importances(self) -> None:
        """Feature importances should be returned correctly."""
        model = MLEStatModel(stat_name="hr")

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
        """Feature importances should raise when not fitted."""
        model = MLEStatModel(stat_name="hr")

        with pytest.raises(ValueError, match="has not been fitted"):
            model.feature_importances()

    def test_custom_hyperparameters(self) -> None:
        """Custom hyperparameters should be used."""
        hp = MLEHyperparameters(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
        )
        model = MLEStatModel(stat_name="hr", hyperparameters=hp)

        assert model.hyperparameters.n_estimators == 50
        assert model.hyperparameters.max_depth == 3
        assert model.hyperparameters.learning_rate == 0.05

    def test_get_params_and_from_params(self) -> None:
        """Model should serialize and deserialize correctly."""
        model = MLEStatModel(stat_name="hr")

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100) * 0.01 + 0.03
        feature_names = ["f1", "f2", "f3"]

        model.fit(X, y, feature_names)
        original_predictions = model.predict(X[:5])

        # Serialize and deserialize
        params = model.get_params()
        restored = MLEStatModel.from_params(params)

        assert restored.stat_name == "hr"
        assert restored.is_fitted
        restored_predictions = restored.predict(X[:5])
        np.testing.assert_array_almost_equal(original_predictions, restored_predictions)

    def test_default_hyperparameters(self) -> None:
        """Default hyperparameters should be reasonable for MLE."""
        hp = MLEHyperparameters()

        assert hp.n_estimators == 100
        assert hp.max_depth == 4
        assert hp.learning_rate == 0.1
        assert hp.min_child_samples == 20
        assert hp.subsample == 0.8
        assert hp.random_state == 42


class TestMLEGradientBoostingModel:
    """Tests for MLEGradientBoostingModel."""

    def test_add_model_and_predict_rates(self) -> None:
        """Should add models and predict rates correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = MLEGradientBoostingModel(
            player_type="batter",
            feature_names=feature_names,
            training_years=(2022, 2023),
        )

        # Train and add models for two stats
        for stat in ["hr", "so"]:
            y = np.random.randn(100) * 0.01 + 0.05
            model = MLEStatModel(stat_name=stat)
            model.fit(X, y, feature_names)
            model_set.add_model(model)

        # Predict rates for a single feature vector
        test_features = X[0]
        rates = model_set.predict_rates(test_features)

        assert "hr" in rates
        assert "so" in rates
        assert isinstance(rates["hr"], float)
        assert isinstance(rates["so"], float)

    def test_predict_rates_batch(self) -> None:
        """Should predict rates for multiple samples."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = MLEGradientBoostingModel(
            player_type="batter",
            feature_names=feature_names,
        )

        model = MLEStatModel(stat_name="hr")
        model.fit(X, np.random.randn(100) * 0.01, feature_names)
        model_set.add_model(model)

        # Predict for multiple samples
        rates = model_set.predict_rates_batch(X[:10])

        assert "hr" in rates
        assert rates["hr"].shape == (10,)

    def test_get_stats(self) -> None:
        """Should return list of fitted stats."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = MLEGradientBoostingModel(player_type="batter")

        # Add one fitted model
        y = np.random.randn(50) * 0.01
        model = MLEStatModel(stat_name="hr")
        model.fit(X, y, feature_names)
        model_set.add_model(model)

        # Add one unfitted model
        unfitted = MLEStatModel(stat_name="so")
        model_set.models["so"] = unfitted

        stats = model_set.get_stats()
        assert stats == ["hr"]

    def test_predict_rates_handles_2d_input(self) -> None:
        """Should handle 2D input (1, n_features)."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = MLEGradientBoostingModel(player_type="batter", feature_names=feature_names)

        model = MLEStatModel(stat_name="hr")
        model.fit(X, np.random.randn(50) * 0.01, feature_names)
        model_set.add_model(model)

        # Test with 2D input (1, n_features)
        test_features = X[0].reshape(1, -1)
        rates = model_set.predict_rates(test_features)

        assert "hr" in rates

    def test_feature_importances(self) -> None:
        """Should return feature importances for specific stat."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 5 + np.random.randn(100) * 0.1
        feature_names = ["f1", "f2", "f3"]

        model_set = MLEGradientBoostingModel(player_type="batter")

        model = MLEStatModel(stat_name="hr")
        model.fit(X, y, feature_names)
        model_set.add_model(model)

        importances = model_set.feature_importances("hr")
        assert set(importances.keys()) == {"f1", "f2", "f3"}

    def test_feature_importances_raises_for_unknown_stat(self) -> None:
        """Should raise KeyError for unknown stat."""
        model_set = MLEGradientBoostingModel(player_type="batter")

        with pytest.raises(KeyError, match="No model for stat"):
            model_set.feature_importances("unknown")

    def test_get_params_and_from_params(self) -> None:
        """Should serialize and deserialize correctly."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = MLEGradientBoostingModel(
            player_type="batter",
            feature_names=feature_names,
            training_years=(2022, 2023),
        )

        model = MLEStatModel(stat_name="hr")
        model.fit(X, np.random.randn(50) * 0.01, feature_names)
        model_set.add_model(model)

        # Serialize and deserialize
        params = model_set.get_params()
        restored = MLEGradientBoostingModel.from_params(params)

        assert restored.player_type == "batter"
        assert restored.feature_names == feature_names
        assert restored.training_years == (2022, 2023)
        assert "hr" in restored.models
        assert restored.models["hr"].is_fitted

    def test_predict_rates_batch_unfitted_returns_zeros(self) -> None:
        """Unfitted stats should return zeros in batch prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        feature_names = ["f1", "f2", "f3"]

        model_set = MLEGradientBoostingModel(player_type="batter")

        # Add fitted model
        model_hr = MLEStatModel(stat_name="hr")
        model_hr.fit(X, np.random.randn(100) * 0.01, feature_names)
        model_set.add_model(model_hr)

        # Add unfitted model
        model_so = MLEStatModel(stat_name="so")
        model_set.models["so"] = model_so

        rates = model_set.predict_rates_batch(X[:5])

        assert rates["hr"].shape == (5,)
        assert rates["so"].shape == (5,)
        np.testing.assert_array_equal(rates["so"], np.zeros(5))


class TestMLEHyperparameters:
    """Tests for MLEHyperparameters dataclass."""

    def test_frozen(self) -> None:
        """Hyperparameters should be frozen (immutable)."""
        hp = MLEHyperparameters()

        with pytest.raises(AttributeError):
            hp.n_estimators = 200  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equal hyperparameters should be equal."""
        hp1 = MLEHyperparameters(n_estimators=50)
        hp2 = MLEHyperparameters(n_estimators=50)

        assert hp1 == hp2

    def test_inequality(self) -> None:
        """Different hyperparameters should not be equal."""
        hp1 = MLEHyperparameters(n_estimators=50)
        hp2 = MLEHyperparameters(n_estimators=100)

        assert hp1 != hp2
