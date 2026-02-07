"""Training orchestration for Multi-Task Learning models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from fantasy_baseball_manager.ml.mtl.config import MTLArchitectureConfig, MTLTrainingConfig
from fantasy_baseball_manager.ml.mtl.dataset import (
    BatterTrainingDataCollector,
    MTLDataset,
    PitcherTrainingDataCollector,
)
from fantasy_baseball_manager.ml.mtl.model import (
    BATTER_STATS,
    PITCHER_STATS,
    MultiTaskBatterModel,
    MultiTaskNet,
    MultiTaskPitcherModel,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallDataSource
    from fantasy_baseball_manager.pipeline.skill_data import SkillDataSource
    from fantasy_baseball_manager.pipeline.statcast_data import FullStatcastDataSource
    from fantasy_baseball_manager.player_id.mapper import SfbbMapper

logger = logging.getLogger(__name__)


@dataclass
class MTLTrainer:
    """Trains MTL models to predict raw stat rates.

    Training flow:
    1. Collect features and actual rates from historical data
    2. Split into train/validation sets
    3. Train neural network with early stopping
    4. Return trained model with validation metrics
    """

    batting_source: DataSource[BattingSeasonStats]
    pitching_source: DataSource[PitchingSeasonStats]
    statcast_source: FullStatcastDataSource
    batted_ball_source: PitcherBattedBallDataSource
    skill_data_source: SkillDataSource
    id_mapper: SfbbMapper
    training_config: MTLTrainingConfig = field(default_factory=MTLTrainingConfig)
    architecture_config: MTLArchitectureConfig = field(default_factory=MTLArchitectureConfig)

    def train_batter_model(
        self,
        target_years: tuple[int, ...],
    ) -> tuple[MultiTaskBatterModel, dict[str, float]]:
        """Train batter model on historical data.

        Args:
            target_years: Years to use as targets (features from year-1).

        Returns:
            Tuple of (trained model, validation metrics dict).
        """
        # Collect training data
        collector = BatterTrainingDataCollector(
            batting_source=self.batting_source,
            statcast_source=self.statcast_source,
            skill_data_source=self.skill_data_source,
            id_mapper=self.id_mapper,
            min_pa=self.training_config.batter_min_pa,
        )

        features, rates, feature_names = collector.collect(target_years)

        if len(features) < self.training_config.min_samples:
            logger.warning(
                "Insufficient samples for batter MTL training: %d < %d",
                len(features),
                self.training_config.min_samples,
            )
            return MultiTaskBatterModel(), {}

        logger.info("Collected %d batter training samples", len(features))

        # Split train/validation
        n_val = max(1, int(len(features) * self.training_config.val_fraction))
        indices = np.random.permutation(len(features))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_features = features[train_indices]
        train_rates = {stat: rates[stat][train_indices] for stat in BATTER_STATS}
        val_features = features[val_indices]
        val_rates = {stat: rates[stat][val_indices] for stat in BATTER_STATS}

        # Create datasets
        train_dataset = MTLDataset(train_features, train_rates)
        val_dataset = MTLDataset(val_features, val_rates)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

        # Create network
        network = MultiTaskNet(
            n_features=len(feature_names),
            target_stats=BATTER_STATS,
            config=self.architecture_config,
        )

        # Train
        val_metrics = self._train_network(network, train_loader, val_loader)

        # Create model wrapper
        model = MultiTaskBatterModel(
            network=network,
            feature_names=feature_names,
            training_years=target_years,
            validation_metrics=val_metrics,
        )
        model._is_fitted = True

        return model, val_metrics

    def train_pitcher_model(
        self,
        target_years: tuple[int, ...],
    ) -> tuple[MultiTaskPitcherModel, dict[str, float]]:
        """Train pitcher model on historical data.

        Args:
            target_years: Years to use as targets (features from year-1).

        Returns:
            Tuple of (trained model, validation metrics dict).
        """
        # Collect training data
        collector = PitcherTrainingDataCollector(
            pitching_source=self.pitching_source,
            statcast_source=self.statcast_source,
            batted_ball_source=self.batted_ball_source,
            id_mapper=self.id_mapper,
            min_pa=self.training_config.pitcher_min_pa,
        )

        features, rates, feature_names = collector.collect(target_years)

        if len(features) < self.training_config.min_samples:
            logger.warning(
                "Insufficient samples for pitcher MTL training: %d < %d",
                len(features),
                self.training_config.min_samples,
            )
            return MultiTaskPitcherModel(), {}

        logger.info("Collected %d pitcher training samples", len(features))

        # Split train/validation
        n_val = max(1, int(len(features) * self.training_config.val_fraction))
        indices = np.random.permutation(len(features))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_features = features[train_indices]
        train_rates = {stat: rates[stat][train_indices] for stat in PITCHER_STATS}
        val_features = features[val_indices]
        val_rates = {stat: rates[stat][val_indices] for stat in PITCHER_STATS}

        # Create datasets
        train_dataset = MTLDataset(train_features, train_rates)
        val_dataset = MTLDataset(val_features, val_rates)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
        )

        # Create network
        network = MultiTaskNet(
            n_features=len(feature_names),
            target_stats=PITCHER_STATS,
            config=self.architecture_config,
        )

        # Train
        val_metrics = self._train_network(network, train_loader, val_loader)

        # Create model wrapper
        model = MultiTaskPitcherModel(
            network=network,
            feature_names=feature_names,
            training_years=target_years,
            validation_metrics=val_metrics,
        )
        model._is_fitted = True

        return model, val_metrics

    def _train_network(
        self,
        network: MultiTaskNet,
        train_loader: DataLoader[tuple[torch.Tensor, dict[str, torch.Tensor]]],
        val_loader: DataLoader[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, float]:
        """Train the network with early stopping.

        Returns:
            Dict of validation metrics (per-stat RMSE).
        """
        module = network.module
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state_dict: dict[str, torch.Tensor] | None = None

        for epoch in range(self.training_config.epochs):
            # Training
            module.train()
            train_loss = 0.0
            for batch_features, batch_rates in train_loader:
                optimizer.zero_grad()
                predictions = module(batch_features)
                loss = network.compute_loss(predictions, batch_rates)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            module.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_rates in val_loader:
                    predictions = module(batch_features)
                    loss = network.compute_loss(predictions, batch_rates)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = {k: v.clone() for k, v in module.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.training_config.patience:
                    logger.info(
                        "Early stopping at epoch %d (best val loss: %.4f)",
                        epoch,
                        best_val_loss,
                    )
                    break

            if epoch % 20 == 0:
                logger.debug(
                    "Epoch %d: train_loss=%.4f, val_loss=%.4f",
                    epoch,
                    train_loss,
                    val_loss,
                )

        # Restore best model
        if best_state_dict is not None:
            module.load_state_dict(best_state_dict)

        # Compute per-stat validation metrics
        val_metrics = self._compute_validation_metrics(network, val_loader)
        logger.info("Validation metrics: %s", val_metrics)

        return val_metrics

    def _compute_validation_metrics(
        self,
        network: MultiTaskNet,
        val_loader: DataLoader[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ) -> dict[str, float]:
        """Compute per-stat RMSE on validation set."""
        module = network.module
        module.eval()

        all_predictions: dict[str, list[float]] = {stat: [] for stat in network.target_stats}
        all_targets: dict[str, list[float]] = {stat: [] for stat in network.target_stats}

        with torch.no_grad():
            for batch_features, batch_rates in val_loader:
                predictions = module(batch_features)
                for stat in network.target_stats:
                    all_predictions[stat].extend(predictions[stat].squeeze().tolist())
                    all_targets[stat].extend(batch_rates[stat].squeeze().tolist())

        metrics = {}
        for stat in network.target_stats:
            preds = np.array(all_predictions[stat])
            targets = np.array(all_targets[stat])
            rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
            metrics[f"{stat}_rmse"] = rmse

        return metrics
