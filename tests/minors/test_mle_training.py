"""Tests for minors/training.py - MLE model training orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from fantasy_baseball_manager.context import init_context, reset_context
from fantasy_baseball_manager.minors.model import MLEHyperparameters
from fantasy_baseball_manager.minors.training import (
    MLEModelTrainer,
    MLETrainingConfig,
)
from fantasy_baseball_manager.minors.training_data import BATTER_TARGET_STATS
from fantasy_baseball_manager.result import Ok


class TestMLETrainingConfig:
    """Tests for MLETrainingConfig."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = MLETrainingConfig()

        assert config.min_samples == 50
        assert config.min_milb_pa == 200
        assert config.min_mlb_pa == 100
        assert config.max_prior_mlb_pa == 200
        assert isinstance(config.hyperparameters, MLEHyperparameters)

    def test_custom_hyperparameters(self) -> None:
        """Config should accept custom hyperparameters."""
        hp = MLEHyperparameters(n_estimators=50, max_depth=3)
        config = MLETrainingConfig(hyperparameters=hp)

        assert config.hyperparameters.n_estimators == 50
        assert config.hyperparameters.max_depth == 3


class TestMLEModelTrainer:
    """Tests for MLEModelTrainer."""

    def setup_method(self) -> None:
        init_context(year=2024)

    def teardown_method(self) -> None:
        reset_context()

    @pytest.fixture
    def mock_milb_source(self) -> MagicMock:
        """Create a mock MiLB data source."""
        return MagicMock()

    @pytest.fixture
    def mock_mlb_source(self) -> MagicMock:
        """Create a mock MLB data source."""
        return MagicMock()

    def test_train_batter_models_empty_data(
        self,
        mock_milb_source: MagicMock,
        mock_mlb_source: MagicMock,
    ) -> None:
        """Should handle empty training data gracefully."""
        # Mock the sources to return empty data when called
        mock_milb_source.return_value = Ok([])
        mock_mlb_source.return_value = Ok([])

        trainer = MLEModelTrainer(
            milb_source=mock_milb_source,
            mlb_batting_source=mock_mlb_source,
        )

        model_set = trainer.train_batter_models(target_years=(2022,))

        assert model_set.player_type == "batter"
        assert model_set.training_years == (2022,)
        assert len(model_set.get_stats()) == 0

    def test_train_batter_models_with_synthetic_data(self) -> None:
        """Should train models with synthetic training data."""
        # Create mock data collector that returns synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 32

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        targets = {stat: np.random.rand(n_samples) * 0.1 for stat in BATTER_TARGET_STATS}
        weights = np.abs(np.random.randn(n_samples)) * 100 + 100
        feature_names = [f"f{i}" for i in range(n_features)]

        # Create mocks
        mock_milb_source = MagicMock()
        mock_mlb_source = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.feature_names.return_value = feature_names

        trainer = MLEModelTrainer(
            milb_source=mock_milb_source,
            mlb_batting_source=mock_mlb_source,
            feature_extractor=mock_extractor,
        )

        # Patch the collector to return synthetic data
        with pytest.MonkeyPatch.context() as mp:

            def mock_collect(_self, _target_years: tuple[int, ...], include_aggregated_stats: bool = False):
                return X, targets, weights, feature_names, None

            mp.setattr(
                "fantasy_baseball_manager.minors.training.MLETrainingDataCollector.collect",
                mock_collect,
            )

            model_set = trainer.train_batter_models(target_years=(2022, 2023))

        assert model_set.player_type == "batter"
        assert model_set.training_years == (2022, 2023)
        # All stats should have models since we have enough samples
        assert len(model_set.get_stats()) == len(BATTER_TARGET_STATS)

    def test_train_batter_models_insufficient_samples(self) -> None:
        """Should skip stats with insufficient samples."""
        np.random.seed(42)
        n_samples = 30  # Below min_samples default of 50
        n_features = 32

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        targets = {stat: np.random.rand(n_samples) * 0.1 for stat in BATTER_TARGET_STATS}
        weights = np.abs(np.random.randn(n_samples)) * 100 + 100
        feature_names = [f"f{i}" for i in range(n_features)]

        mock_milb_source = MagicMock()
        mock_mlb_source = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.feature_names.return_value = feature_names

        trainer = MLEModelTrainer(
            milb_source=mock_milb_source,
            mlb_batting_source=mock_mlb_source,
            feature_extractor=mock_extractor,
            config=MLETrainingConfig(min_samples=50),
        )

        with pytest.MonkeyPatch.context() as mp:

            def mock_collect(_self, _target_years: tuple[int, ...], include_aggregated_stats: bool = False):
                return X, targets, weights, feature_names, None

            mp.setattr(
                "fantasy_baseball_manager.minors.training.MLETrainingDataCollector.collect",
                mock_collect,
            )

            model_set = trainer.train_batter_models(target_years=(2022,))

        # No stats should have models since we have insufficient samples
        assert len(model_set.get_stats()) == 0

    def test_train_batter_models_with_validation(self) -> None:
        """Should use validation data for early stopping when provided."""
        np.random.seed(42)
        n_train = 100
        n_val = 30
        n_features = 32

        X_train = np.random.randn(n_train, n_features).astype(np.float32)
        X_val = np.random.randn(n_val, n_features).astype(np.float32)
        targets_train = {stat: np.random.rand(n_train) * 0.1 for stat in BATTER_TARGET_STATS}
        targets_val = {stat: np.random.rand(n_val) * 0.1 for stat in BATTER_TARGET_STATS}
        weights_train = np.abs(np.random.randn(n_train)) * 100 + 100
        weights_val = np.abs(np.random.randn(n_val)) * 100 + 100
        feature_names = [f"f{i}" for i in range(n_features)]

        mock_milb_source = MagicMock()
        mock_mlb_source = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.feature_names.return_value = feature_names

        trainer = MLEModelTrainer(
            milb_source=mock_milb_source,
            mlb_batting_source=mock_mlb_source,
            feature_extractor=mock_extractor,
        )

        call_count = [0]

        def mock_collect(_self, _target_years: tuple[int, ...], include_aggregated_stats: bool = False):
            call_count[0] += 1
            if call_count[0] == 1:
                return X_train, targets_train, weights_train, feature_names, None
            else:
                return X_val, targets_val, weights_val, feature_names, None

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "fantasy_baseball_manager.minors.training.MLETrainingDataCollector.collect",
                mock_collect,
            )

            model_set = trainer.train_batter_models(
                target_years=(2022,),
                validation_years=(2023,),
                early_stopping_rounds=10,
            )

        assert model_set.player_type == "batter"
        assert len(model_set.get_stats()) == len(BATTER_TARGET_STATS)

    def test_custom_config(self) -> None:
        """Should use custom config values."""
        hp = MLEHyperparameters(n_estimators=50)
        config = MLETrainingConfig(
            min_samples=100,
            min_milb_pa=300,
            hyperparameters=hp,
        )

        mock_milb_source = MagicMock()
        mock_mlb_source = MagicMock()

        trainer = MLEModelTrainer(
            milb_source=mock_milb_source,
            mlb_batting_source=mock_mlb_source,
            config=config,
        )

        assert trainer.config.min_samples == 100
        assert trainer.config.min_milb_pa == 300
        assert trainer.config.hyperparameters.n_estimators == 50
