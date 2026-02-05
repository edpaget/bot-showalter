"""Training orchestration for MLE gradient boosting models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.minors.model import (
    MLEGradientBoostingModel,
    MLEHyperparameters,
    MLEStatModel,
)
from fantasy_baseball_manager.minors.training_data import (
    BATTER_TARGET_STATS,
    MLETrainingDataCollector,
)

if TYPE_CHECKING:
    import numpy as np

    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.minors.data_source import MinorLeagueDataSource
    from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor
    from fantasy_baseball_manager.minors.types import MiLBStatcastStats

logger = logging.getLogger(__name__)


@dataclass
class MLETrainingConfig:
    """Configuration for MLE model training."""

    min_samples: int = 50
    hyperparameters: MLEHyperparameters = field(default_factory=MLEHyperparameters)
    min_milb_pa: int = 200
    min_mlb_pa: int = 100
    max_prior_mlb_pa: int = 200


@dataclass
class MLEModelTrainer:
    """Orchestrates training MLE models from historical call-up data.

    Training data flow (for target_year=2022):
    1. Get MiLB stats from 2021 (year-1) for players at AAA/AA
    2. Aggregate stats across levels (PA-weighted)
    3. Get MLB stats from 2022 (or 2023 for late call-ups)
    4. Extract features from MiLB data
    5. Compute target rates from MLB data
    6. Train model: MiLB features → MLB rates
    """

    milb_source: MinorLeagueDataSource
    mlb_source: StatsDataSource
    config: MLETrainingConfig = field(default_factory=MLETrainingConfig)
    feature_extractor: MLEBatterFeatureExtractor | None = None
    statcast_lookup: dict[tuple[str, int], MiLBStatcastStats] | None = None

    def train_batter_models(
        self,
        target_years: tuple[int, ...],
        validation_years: tuple[int, ...] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> MLEGradientBoostingModel:
        """Train MLE batter models using data from target years.

        Args:
            target_years: Years to use as MLB targets (e.g., (2022, 2023))
                          MiLB features come from year-1, MLB outcomes from target year.
            validation_years: Optional years to use for validation/early stopping.
            early_stopping_rounds: Stop training if no improvement for this many rounds.

        Returns:
            Trained MLEGradientBoostingModel for batters
        """
        # Create training data collector
        collector = MLETrainingDataCollector(
            milb_source=self.milb_source,
            mlb_source=self.mlb_source,
            min_milb_pa=self.config.min_milb_pa,
            min_mlb_pa=self.config.min_mlb_pa,
            max_prior_mlb_pa=self.config.max_prior_mlb_pa,
            feature_extractor=self.feature_extractor,
            statcast_lookup=self.statcast_lookup,
        )

        # Collect training data
        logger.info("Collecting training data for years %s", target_years)
        X_train, y_train, w_train, feature_names = collector.collect(target_years)

        if len(X_train) == 0:
            logger.warning("No training samples collected")
            return MLEGradientBoostingModel(
                player_type="batter",
                feature_names=feature_names,
                training_years=target_years,
            )

        logger.info("Collected %d training samples with %d features", len(X_train), len(feature_names))

        # Collect validation data if provided
        X_val: np.ndarray | None = None
        y_val: dict[str, np.ndarray] | None = None
        if validation_years is not None:
            logger.info("Collecting validation data for years %s", validation_years)
            X_val_collected, y_val_collected, _, _ = collector.collect(validation_years)
            if len(X_val_collected) > 0:
                X_val = X_val_collected
                y_val = y_val_collected
                logger.info("Collected %d validation samples", len(X_val_collected))
            else:
                logger.warning("No validation samples collected, disabling early stopping")

        # Train models
        model_set = MLEGradientBoostingModel(
            player_type="batter",
            feature_names=feature_names,
            training_years=target_years,
        )

        for stat in BATTER_TARGET_STATS:
            y_stat = y_train[stat]

            if len(y_stat) < self.config.min_samples:
                logger.warning(
                    "Insufficient samples for batter %s model: %d < %d",
                    stat,
                    len(y_stat),
                    self.config.min_samples,
                )
                continue

            model = MLEStatModel(
                stat_name=stat,
                hyperparameters=self.config.hyperparameters,
            )

            # Prepare validation set if available
            eval_set: tuple[np.ndarray, np.ndarray] | None = None
            if X_val is not None and y_val is not None and early_stopping_rounds is not None:
                eval_set = (X_val, y_val[stat])

            model.fit(
                X_train,
                y_stat,
                feature_names,
                sample_weight=w_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
            )
            model_set.add_model(model)
            logger.info("Trained batter %s model on %d samples", stat, len(y_stat))

        return model_set
