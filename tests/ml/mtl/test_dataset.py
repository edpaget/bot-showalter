"""Tests for MTL dataset classes."""

import numpy as np
import torch

from fantasy_baseball_manager.ml.mtl.dataset import MTLDataset
from fantasy_baseball_manager.ml.mtl.model import BATTER_STATS


class TestMTLDataset:
    def test_init(self) -> None:
        """Test dataset initialization."""
        n_samples = 100
        n_features = 25
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        rates = {stat: np.random.rand(n_samples).astype(np.float32) for stat in BATTER_STATS}

        dataset = MTLDataset(features, rates)

        assert len(dataset) == n_samples

    def test_getitem(self) -> None:
        """Test getting a single item."""
        n_samples = 10
        n_features = 5
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        rates = {"hr": np.random.rand(n_samples).astype(np.float32)}

        dataset = MTLDataset(features, rates)
        item_features, item_rates = dataset[0]

        assert item_features.shape == (n_features,)
        assert "hr" in item_rates
        assert item_rates["hr"].shape == (1,)

    def test_getitem_all_stats(self) -> None:
        """Test that all stats are returned in rates dict."""
        features = np.random.randn(5, 10).astype(np.float32)
        rates = {
            "hr": np.random.rand(5).astype(np.float32),
            "so": np.random.rand(5).astype(np.float32),
            "bb": np.random.rand(5).astype(np.float32),
        }

        dataset = MTLDataset(features, rates)
        _, item_rates = dataset[2]

        assert set(item_rates.keys()) == {"hr", "so", "bb"}

    def test_tensor_types(self) -> None:
        """Test that tensors are float32."""
        features = np.random.randn(5, 10).astype(np.float64)  # Note: float64
        rates = {"hr": np.random.rand(5).astype(np.float64)}

        dataset = MTLDataset(features, rates)
        item_features, item_rates = dataset[0]

        assert item_features.dtype == torch.float32
        assert item_rates["hr"].dtype == torch.float32

    def test_works_with_dataloader(self) -> None:
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        features = np.random.randn(20, 10).astype(np.float32)
        rates = {"hr": np.random.rand(20).astype(np.float32)}

        dataset = MTLDataset(features, rates)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch_features, batch_rates = next(iter(loader))

        assert batch_features.shape == (4, 10)
        assert batch_rates["hr"].shape == (4, 1)
