"""Tests for MTL trainer module."""

import numpy as np
import torch

from fantasy_baseball_manager.ml.mtl.config import MTLArchitectureConfig, MTLTrainingConfig
from fantasy_baseball_manager.ml.mtl.dataset import MTLDataset
from fantasy_baseball_manager.ml.mtl.model import MultiTaskNet


class TestMTLTrainerInternal:
    """Test internal trainer components without external data sources."""

    def test_training_loop_converges(self) -> None:
        """Test that the training loop reduces loss."""
        from torch.utils.data import DataLoader

        # Create synthetic data with learnable pattern
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features).astype(np.float32)

        # Create target rates that have some relationship to features
        rates = {
            "hr": (X[:, 0] * 0.01 + 0.03).astype(np.float32),  # HR rate ~3%
            "so": (X[:, 1] * 0.02 + 0.20).astype(np.float32),  # SO rate ~20%
        }

        # Split train/val
        n_val = 15
        train_dataset = MTLDataset(X[n_val:], {k: v[n_val:] for k, v in rates.items()})
        val_dataset = MTLDataset(X[:n_val], {k: v[:n_val] for k, v in rates.items()})

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Create network
        network = MultiTaskNet(
            n_features=n_features,
            target_stats=("hr", "so"),
            config=MTLArchitectureConfig(
                shared_layers=(16, 8),
                head_hidden_size=4,
            ),
        )

        # Train for a few epochs
        module = network.module
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        for _epoch in range(20):
            module.train()
            epoch_loss = 0.0
            for batch_features, batch_rates in train_loader:
                optimizer.zero_grad()
                predictions = module(batch_features)
                loss = network.compute_loss(predictions, batch_rates)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if initial_loss is None:
                initial_loss = epoch_loss
            final_loss = epoch_loss

        assert final_loss is not None
        assert initial_loss is not None
        assert final_loss < initial_loss, "Training should reduce loss"

    def test_validation_metrics_computed(self) -> None:
        """Test that validation metrics are computed correctly."""
        # Create simple data
        np.random.seed(42)
        X = np.random.randn(20, 5).astype(np.float32)
        rates = {"hr": np.random.rand(20).astype(np.float32) * 0.1}

        val_dataset = MTLDataset(X, rates)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

        network = MultiTaskNet(n_features=5, target_stats=("hr",))
        module = network.module
        module.eval()

        # Collect predictions
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_features, batch_rates in val_loader:
                predictions = module(batch_features)
                all_preds.extend(predictions["hr"].squeeze().tolist())
                all_targets.extend(batch_rates["hr"].squeeze().tolist())

        preds = np.array(all_preds)
        targets = np.array(all_targets)
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

        assert rmse > 0  # RMSE should be positive
        assert rmse < 1.0  # For rates between 0-0.1, RMSE should be small

    def test_early_stopping_logic(self) -> None:
        """Test that early stopping logic triggers correctly."""
        # Test the early stopping logic directly without training
        # by simulating validation losses that plateau

        patience = 3
        best_val_loss = float("inf")
        patience_counter = 0
        stopped_early = False

        # Simulate: losses decrease for 5 epochs, then plateau
        val_losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.45, 0.46, 0.45, 0.47, 0.48]

        for epoch, val_loss in enumerate(val_losses):  # noqa: B007 - epoch used in assertion
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stopped_early = True
                    break

        # Should stop after patience is exceeded (epoch 7)
        assert stopped_early
        assert epoch == 7  # 0-indexed: losses[0:8] processed


class TestMTLTrainingConfig:
    def test_training_config_defaults(self) -> None:
        """Test that training config has sensible defaults."""
        config = MTLTrainingConfig()
        assert config.epochs > 0
        assert config.batch_size > 0
        assert 0 < config.learning_rate < 1
        assert 0 < config.val_fraction < 1
        assert config.patience > 0
