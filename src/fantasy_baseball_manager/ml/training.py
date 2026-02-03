"""Training orchestration for residual prediction models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor
from fantasy_baseball_manager.ml.residual_model import (
    ModelHyperparameters,
    ResidualModelSet,
    StatResidualModel,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallDataSource
    from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
    from fantasy_baseball_manager.pipeline.statcast_data import FullStatcastDataSource
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)

# Stats we train models for
BATTER_STATS = ("hr", "so", "bb", "singles", "doubles", "triples", "sb")
PITCHER_STATS = ("h", "er", "so", "bb", "hr")


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    min_samples: int = 50
    hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    batter_min_pa: int = 100
    pitcher_min_pa: int = 100


@dataclass
class ResidualModelTrainer:
    """Orchestrates training residual models across historical years.

    Training data flow (for target_year=2022):
    1. Generate Marcel projections for 2022 (using 2019-2021 stats)
    2. Extract features from 2021 Statcast data (most recent available)
    3. Get 2022 actuals
    4. Compute residuals: actual_2022 - projected_2022
    5. Train model: features → residuals
    """

    pipeline: ProjectionPipeline
    data_source: StatsDataSource
    statcast_source: FullStatcastDataSource
    batted_ball_source: PitcherBattedBallDataSource
    id_mapper: PlayerIdMapper
    config: TrainingConfig = field(default_factory=TrainingConfig)

    def train_batter_models(self, target_years: tuple[int, ...]) -> ResidualModelSet:
        """Train batter residual models using data from target years.

        Args:
            target_years: Years to use as targets (e.g., (2020, 2021, 2022))
                          Features come from year-1, actuals from target year.

        Returns:
            Trained ResidualModelSet for batters
        """
        extractor = BatterFeatureExtractor(min_pa=self.config.batter_min_pa)
        feature_names = extractor.feature_names()

        # Collect training data across years
        all_features: dict[str, list[np.ndarray]] = {stat: [] for stat in BATTER_STATS}
        all_residuals: dict[str, list[float]] = {stat: [] for stat in BATTER_STATS}

        for target_year in target_years:
            logger.info("Processing batter training data for target year %d", target_year)

            # Get projections for target year
            projections = self.pipeline.project_batters(self.data_source, target_year)
            proj_lookup = {p.player_id: p for p in projections}

            # Get actuals for target year
            actuals = self.data_source.batting_stats(target_year)
            actual_lookup = {a.player_id: a for a in actuals}

            # Get Statcast data from prior year (most recent available when making projection)
            statcast_year = target_year - 1
            statcast_data = self.statcast_source.batter_expected_stats(statcast_year)

            # Build MLBAM -> Statcast lookup
            statcast_lookup = {s.player_id: s for s in statcast_data}

            # Process each player with available data
            for fg_id, proj in proj_lookup.items():
                actual = actual_lookup.get(fg_id)
                if actual is None or actual.pa < self.config.batter_min_pa:
                    continue

                # Map FanGraphs ID to MLBAM for Statcast lookup
                mlbam_id = self.id_mapper.fangraphs_to_mlbam(fg_id)
                if mlbam_id is None:
                    continue

                statcast = statcast_lookup.get(mlbam_id)
                if statcast is None:
                    continue

                # Create a PlayerRates-like object for feature extraction
                from fantasy_baseball_manager.pipeline.types import PlayerRates

                player_rates = PlayerRates(
                    player_id=fg_id,
                    name=proj.name,
                    year=target_year,
                    age=proj.age,
                    rates={
                        "hr": proj.hr / proj.pa if proj.pa > 0 else 0,
                        "so": proj.so / proj.pa if proj.pa > 0 else 0,
                        "bb": proj.bb / proj.pa if proj.pa > 0 else 0,
                        "singles": proj.singles / proj.pa if proj.pa > 0 else 0,
                        "doubles": proj.doubles / proj.pa if proj.pa > 0 else 0,
                        "triples": proj.triples / proj.pa if proj.pa > 0 else 0,
                        "sb": proj.sb / proj.pa if proj.pa > 0 else 0,
                    },
                    opportunities=proj.pa,
                )

                features = extractor.extract(player_rates, statcast)
                if features is None:
                    continue

                # Compute residuals for each stat (counting stat residual)
                for stat in BATTER_STATS:
                    projected_count = getattr(proj, stat, 0)
                    actual_count = getattr(actual, stat, 0)
                    residual = actual_count - projected_count
                    all_features[stat].append(features)
                    all_residuals[stat].append(residual)

        # Train models
        model_set = ResidualModelSet(
            player_type="batter",
            feature_names=feature_names,
            training_years=target_years,
        )

        for stat in BATTER_STATS:
            if len(all_features[stat]) < self.config.min_samples:
                logger.warning(
                    "Insufficient samples for batter %s model: %d < %d",
                    stat,
                    len(all_features[stat]),
                    self.config.min_samples,
                )
                continue

            X = np.array(all_features[stat])
            y = np.array(all_residuals[stat])

            model = StatResidualModel(stat_name=stat, hyperparameters=self.config.hyperparameters)
            model.fit(X, y, feature_names)
            model_set.add_model(model)
            logger.info("Trained batter %s model on %d samples", stat, len(y))

        return model_set

    def train_pitcher_models(self, target_years: tuple[int, ...]) -> ResidualModelSet:
        """Train pitcher residual models using data from target years.

        Args:
            target_years: Years to use as targets (e.g., (2020, 2021, 2022))
                          Features come from year-1, actuals from target year.

        Returns:
            Trained ResidualModelSet for pitchers
        """
        extractor = PitcherFeatureExtractor(min_pa=self.config.pitcher_min_pa)
        feature_names = extractor.feature_names()

        # Collect training data across years
        all_features: dict[str, list[np.ndarray]] = {stat: [] for stat in PITCHER_STATS}
        all_residuals: dict[str, list[float]] = {stat: [] for stat in PITCHER_STATS}

        for target_year in target_years:
            logger.info("Processing pitcher training data for target year %d", target_year)

            # Get projections for target year
            projections = self.pipeline.project_pitchers(self.data_source, target_year)
            proj_lookup = {p.player_id: p for p in projections}

            # Get actuals for target year
            actuals = self.data_source.pitching_stats(target_year)
            actual_lookup = {a.player_id: a for a in actuals}

            # Get Statcast and batted ball data from prior year
            statcast_year = target_year - 1
            statcast_data = self.statcast_source.pitcher_expected_stats(statcast_year)
            batted_ball_data = self.batted_ball_source.pitcher_batted_ball_stats(statcast_year)

            # Build lookups
            statcast_lookup = {s.player_id: s for s in statcast_data}
            bb_lookup = {b.player_id: b for b in batted_ball_data}

            # Process each player with available data
            for fg_id, proj in proj_lookup.items():
                actual = actual_lookup.get(fg_id)
                if actual is None:
                    continue

                # Minimum IP check (convert to outs equivalent)
                actual_outs = int(actual.ip * 3)
                if actual_outs < self.config.pitcher_min_pa:
                    continue

                # Map FanGraphs ID to MLBAM for Statcast lookup
                mlbam_id = self.id_mapper.fangraphs_to_mlbam(fg_id)
                if mlbam_id is None:
                    continue

                statcast = statcast_lookup.get(mlbam_id)
                if statcast is None:
                    continue

                # Batted ball uses FanGraphs ID
                batted_ball = bb_lookup.get(fg_id)

                # Create a PlayerRates-like object for feature extraction
                from fantasy_baseball_manager.pipeline.types import PlayerRates

                proj_outs = proj.ip * 3
                player_rates = PlayerRates(
                    player_id=fg_id,
                    name=proj.name,
                    year=target_year,
                    age=proj.age,
                    rates={
                        "h": proj.h / proj_outs if proj_outs > 0 else 0,
                        "er": proj.er / proj_outs if proj_outs > 0 else 0,
                        "so": proj.so / proj_outs if proj_outs > 0 else 0,
                        "bb": proj.bb / proj_outs if proj_outs > 0 else 0,
                        "hr": proj.hr / proj_outs if proj_outs > 0 else 0,
                    },
                    opportunities=proj_outs,
                    metadata={"is_starter": proj.gs > proj.g / 2 if proj.g > 0 else False},
                )

                features = extractor.extract(player_rates, statcast, batted_ball)
                if features is None:
                    continue

                # Compute residuals for each stat (counting stat residual)
                for stat in PITCHER_STATS:
                    projected_count = getattr(proj, stat, 0)
                    actual_count = getattr(actual, stat, 0)
                    residual = actual_count - projected_count
                    all_features[stat].append(features)
                    all_residuals[stat].append(residual)

        # Train models
        model_set = ResidualModelSet(
            player_type="pitcher",
            feature_names=feature_names,
            training_years=target_years,
        )

        for stat in PITCHER_STATS:
            if len(all_features[stat]) < self.config.min_samples:
                logger.warning(
                    "Insufficient samples for pitcher %s model: %d < %d",
                    stat,
                    len(all_features[stat]),
                    self.config.min_samples,
                )
                continue

            X = np.array(all_features[stat])
            y = np.array(all_residuals[stat])

            model = StatResidualModel(stat_name=stat, hyperparameters=self.config.hyperparameters)
            model.fit(X, y, feature_names)
            model_set.add_model(model)
            logger.info("Trained pitcher %s model on %d samples", stat, len(y))

        return model_set
