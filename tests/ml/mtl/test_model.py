"""Tests for MTL neural network models."""

import numpy as np
import pytest
import torch

from fantasy_baseball_manager.ml.mtl.config import MTLArchitectureConfig
from fantasy_baseball_manager.ml.mtl.model import (
    BATTER_STATS,
    PITCHER_STATS,
    MultiTaskBatterModel,
    MultiTaskNet,
    MultiTaskPitcherModel,
)


class TestMultiTaskNet:
    def test_forward_pass_shape(self) -> None:
        """Test that forward pass produces correct output shapes."""
        n_features = 25
        batch_size = 8
        stats = ("hr", "so", "bb")

        network = MultiTaskNet(n_features=n_features, target_stats=stats)
        x = torch.randn(batch_size, n_features)

        outputs = network.forward(x)

        assert set(outputs.keys()) == set(stats)
        for stat, output in outputs.items():
            assert output.shape == (batch_size, 1), f"{stat} has wrong shape"

    def test_predict_single_sample(self) -> None:
        """Test prediction for a single feature vector."""
        n_features = 10
        stats = ("hr", "so")

        network = MultiTaskNet(n_features=n_features, target_stats=stats)
        features = np.random.randn(n_features).astype(np.float32)

        predictions = network.predict(features)

        assert set(predictions.keys()) == set(stats)
        for stat, pred in predictions.items():
            assert isinstance(pred, float), f"{stat} should be a float"

    def test_predict_with_2d_input(self) -> None:
        """Test prediction with 2D input (1, n_features)."""
        n_features = 10
        stats = ("hr",)

        network = MultiTaskNet(n_features=n_features, target_stats=stats)
        features = np.random.randn(1, n_features).astype(np.float32)

        predictions = network.predict(features)
        assert "hr" in predictions

    def test_custom_architecture(self) -> None:
        """Test network with custom architecture config."""
        config = MTLArchitectureConfig(
            shared_layers=(32, 16),
            head_hidden_size=8,
            dropout_rates=(0.2, 0.1),
            use_batch_norm=False,
        )
        network = MultiTaskNet(
            n_features=10,
            target_stats=("hr", "so"),
            config=config,
        )

        x = torch.randn(4, 10)
        outputs = network.forward(x)

        assert "hr" in outputs
        assert "so" in outputs

    def test_compute_loss(self) -> None:
        """Test loss computation."""
        network = MultiTaskNet(n_features=10, target_stats=("hr", "so"))
        x = torch.randn(8, 10)

        predictions = network.forward(x)
        targets = {stat: torch.randn(8, 1) for stat in ("hr", "so")}

        loss = network.compute_loss(predictions, targets)

        assert loss.ndim == 0  # Scalar
        assert loss.requires_grad

    def test_uncertainty_weights(self) -> None:
        """Test that uncertainty weights are learnable."""
        network = MultiTaskNet(n_features=10, target_stats=("hr", "so", "bb"))

        weights = network.get_uncertainty_weights()

        assert set(weights.keys()) == {"hr", "so", "bb"}
        # Initial weights should be close to 1.0 (log_var initialized to 0)
        for w in weights.values():
            assert 0.5 < w < 2.0

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        network = MultiTaskNet(n_features=10, target_stats=("hr",))
        x = torch.randn(4, 10, requires_grad=True)

        predictions = network.forward(x)
        targets = {"hr": torch.randn(4, 1)}
        loss = network.compute_loss(predictions, targets)
        loss.backward()

        # Check that module parameters have gradients
        for param in network.module.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMultiTaskBatterModel:
    def test_stats_constant(self) -> None:
        """Verify batter stats constant."""
        assert MultiTaskBatterModel.STATS == BATTER_STATS
        assert len(BATTER_STATS) == 7

    def test_not_fitted_initially(self) -> None:
        """Model should not be fitted when first created."""
        model = MultiTaskBatterModel()
        assert not model.is_fitted

    def test_predict_raises_when_not_fitted(self) -> None:
        """Predict should raise ValueError when model not fitted."""
        model = MultiTaskBatterModel()
        features = np.random.randn(25).astype(np.float32)

        with pytest.raises(ValueError, match="has not been fitted"):
            model.predict(features)

    def test_get_params_and_from_params(self) -> None:
        """Test serialization round-trip."""
        network = MultiTaskNet(
            n_features=25,
            target_stats=MultiTaskBatterModel.STATS,
        )
        model = MultiTaskBatterModel(
            network=network,
            feature_names=["f" + str(i) for i in range(25)],
            training_years=(2020, 2021, 2022),
            validation_metrics={"hr_rmse": 0.01, "so_rmse": 0.02},
        )
        model._is_fitted = True

        params = model.get_params()
        restored = MultiTaskBatterModel.from_params(params)

        assert restored.feature_names == model.feature_names
        assert restored.training_years == model.training_years
        assert restored.validation_metrics == model.validation_metrics
        assert restored.is_fitted

    def test_predict_after_restore(self) -> None:
        """Test that restored model can predict."""
        network = MultiTaskNet(
            n_features=25,
            target_stats=MultiTaskBatterModel.STATS,
        )
        model = MultiTaskBatterModel(
            network=network,
            feature_names=["f" + str(i) for i in range(25)],
        )
        model._is_fitted = True

        features = np.random.randn(25).astype(np.float32)
        original_preds = model.predict(features)

        params = model.get_params()
        restored = MultiTaskBatterModel.from_params(params)
        restored_preds = restored.predict(features)

        for stat in MultiTaskBatterModel.STATS:
            assert stat in original_preds
            assert stat in restored_preds
            # Predictions should be identical after restore
            np.testing.assert_almost_equal(original_preds[stat], restored_preds[stat])


class TestMultiTaskPitcherModel:
    def test_stats_constant(self) -> None:
        """Verify pitcher stats constant."""
        assert MultiTaskPitcherModel.STATS == PITCHER_STATS
        assert len(PITCHER_STATS) == 5

    def test_not_fitted_initially(self) -> None:
        """Model should not be fitted when first created."""
        model = MultiTaskPitcherModel()
        assert not model.is_fitted

    def test_get_params_and_from_params(self) -> None:
        """Test serialization round-trip."""
        network = MultiTaskNet(
            n_features=21,
            target_stats=MultiTaskPitcherModel.STATS,
        )
        model = MultiTaskPitcherModel(
            network=network,
            feature_names=["f" + str(i) for i in range(21)],
            training_years=(2021, 2022),
        )
        model._is_fitted = True

        params = model.get_params()
        restored = MultiTaskPitcherModel.from_params(params)

        assert restored.feature_names == model.feature_names
        assert restored.training_years == model.training_years
        assert restored.is_fitted
