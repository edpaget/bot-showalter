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
from fantasy_baseball_manager.ml.validation import (
    EarlyStoppingConfig,
    StatValidationResult,
    ValidationMetrics,
    ValidationReport,
    ValidationStrategy,
    compute_validation_metrics,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallDataSource
    from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
    from fantasy_baseball_manager.pipeline.skill_data import SkillDataSource
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
    skill_data_source: SkillDataSource
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

            # Get skill data from prior year (uses FanGraphs ID)
            skill_data = self.skill_data_source.batter_skill_stats(statcast_year)
            skill_lookup = {s.player_id: s for s in skill_data}

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

                # Get skill data (optional, uses FanGraphs ID)
                player_skill_data = skill_lookup.get(fg_id)

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

                features = extractor.extract(player_rates, statcast, player_skill_data)
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

    def validate_batter_models(
        self,
        target_years: tuple[int, ...],
        strategy: ValidationStrategy,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> ValidationReport:
        """Run validation using specified strategy.

        Args:
            target_years: Years available for training and validation.
            strategy: Validation strategy (e.g., TimeSeriesHoldout, LeaveOneYearOut).
            early_stopping: Optional early stopping configuration.

        Returns:
            ValidationReport with metrics for each stat and fold.
        """
        extractor = BatterFeatureExtractor(min_pa=self.config.batter_min_pa)
        feature_names = extractor.feature_names()

        splits = strategy.generate_splits(target_years)
        stat_results: list[StatValidationResult] = []

        # Compute holdout years (union of all validation years)
        all_holdout_years: set[int] = set()
        for split in splits:
            all_holdout_years.update(split.val_years)

        for stat in BATTER_STATS:
            fold_metrics: list[ValidationMetrics] = []

            for split in splits:
                logger.info(
                    "Validating batter %s: train=%s, val=%s",
                    stat,
                    split.train_years,
                    split.val_years,
                )

                # Collect training data
                train_features, train_residuals = self._collect_batter_data(
                    split.train_years, extractor, stat
                )

                if len(train_features) < self.config.min_samples:
                    logger.warning(
                        "Insufficient training samples for batter %s: %d < %d",
                        stat,
                        len(train_features),
                        self.config.min_samples,
                    )
                    continue

                # Collect validation data
                val_features, val_residuals = self._collect_batter_data(
                    split.val_years, extractor, stat
                )

                if len(val_features) == 0:
                    logger.warning("No validation samples for batter %s", stat)
                    continue

                X_train = np.array(train_features)
                y_train = np.array(train_residuals)
                X_val = np.array(val_features)
                y_val = np.array(val_residuals)

                # Train model
                model = StatResidualModel(
                    stat_name=stat, hyperparameters=self.config.hyperparameters
                )

                if early_stopping is not None and early_stopping.enabled:
                    # Split off a portion of training data for early stopping
                    n_eval = max(1, int(len(X_train) * early_stopping.eval_fraction))
                    X_eval = X_train[-n_eval:]
                    y_eval = y_train[-n_eval:]
                    X_train_es = X_train[:-n_eval]
                    y_train_es = y_train[:-n_eval]

                    model.fit(
                        X_train_es,
                        y_train_es,
                        feature_names,
                        eval_set=(X_eval, y_eval),
                        early_stopping_rounds=early_stopping.patience,
                    )
                else:
                    model.fit(X_train, y_train, feature_names)

                # Predict and compute metrics
                y_pred = model.predict(X_val)
                metrics = compute_validation_metrics(y_val, y_pred, split.fold_name)
                fold_metrics.append(metrics)

            if fold_metrics:
                stat_results.append(
                    StatValidationResult(stat_name=stat, fold_metrics=tuple(fold_metrics))
                )

        # Compute training years (union of all training years)
        all_train_years: set[int] = set()
        for split in splits:
            all_train_years.update(split.train_years)

        return ValidationReport(
            player_type="batter",
            strategy_name=strategy.name,
            stat_results=tuple(stat_results),
            training_years=tuple(sorted(all_train_years)),
            holdout_years=tuple(sorted(all_holdout_years)),
        )

    def validate_pitcher_models(
        self,
        target_years: tuple[int, ...],
        strategy: ValidationStrategy,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> ValidationReport:
        """Run validation using specified strategy.

        Args:
            target_years: Years available for training and validation.
            strategy: Validation strategy (e.g., TimeSeriesHoldout, LeaveOneYearOut).
            early_stopping: Optional early stopping configuration.

        Returns:
            ValidationReport with metrics for each stat and fold.
        """
        extractor = PitcherFeatureExtractor(min_pa=self.config.pitcher_min_pa)
        feature_names = extractor.feature_names()

        splits = strategy.generate_splits(target_years)
        stat_results: list[StatValidationResult] = []

        # Compute holdout years (union of all validation years)
        all_holdout_years: set[int] = set()
        for split in splits:
            all_holdout_years.update(split.val_years)

        for stat in PITCHER_STATS:
            fold_metrics: list[ValidationMetrics] = []

            for split in splits:
                logger.info(
                    "Validating pitcher %s: train=%s, val=%s",
                    stat,
                    split.train_years,
                    split.val_years,
                )

                # Collect training data
                train_features, train_residuals = self._collect_pitcher_data(
                    split.train_years, extractor, stat
                )

                if len(train_features) < self.config.min_samples:
                    logger.warning(
                        "Insufficient training samples for pitcher %s: %d < %d",
                        stat,
                        len(train_features),
                        self.config.min_samples,
                    )
                    continue

                # Collect validation data
                val_features, val_residuals = self._collect_pitcher_data(
                    split.val_years, extractor, stat
                )

                if len(val_features) == 0:
                    logger.warning("No validation samples for pitcher %s", stat)
                    continue

                X_train = np.array(train_features)
                y_train = np.array(train_residuals)
                X_val = np.array(val_features)
                y_val = np.array(val_residuals)

                # Train model
                model = StatResidualModel(
                    stat_name=stat, hyperparameters=self.config.hyperparameters
                )

                if early_stopping is not None and early_stopping.enabled:
                    # Split off a portion of training data for early stopping
                    n_eval = max(1, int(len(X_train) * early_stopping.eval_fraction))
                    X_eval = X_train[-n_eval:]
                    y_eval = y_train[-n_eval:]
                    X_train_es = X_train[:-n_eval]
                    y_train_es = y_train[:-n_eval]

                    model.fit(
                        X_train_es,
                        y_train_es,
                        feature_names,
                        eval_set=(X_eval, y_eval),
                        early_stopping_rounds=early_stopping.patience,
                    )
                else:
                    model.fit(X_train, y_train, feature_names)

                # Predict and compute metrics
                y_pred = model.predict(X_val)
                metrics = compute_validation_metrics(y_val, y_pred, split.fold_name)
                fold_metrics.append(metrics)

            if fold_metrics:
                stat_results.append(
                    StatValidationResult(stat_name=stat, fold_metrics=tuple(fold_metrics))
                )

        # Compute training years (union of all training years)
        all_train_years: set[int] = set()
        for split in splits:
            all_train_years.update(split.train_years)

        return ValidationReport(
            player_type="pitcher",
            strategy_name=strategy.name,
            stat_results=tuple(stat_results),
            training_years=tuple(sorted(all_train_years)),
            holdout_years=tuple(sorted(all_holdout_years)),
        )

    def _collect_batter_data(
        self,
        years: tuple[int, ...],
        extractor: BatterFeatureExtractor,
        stat: str,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Collect features and residuals for specified years.

        Args:
            years: Target years to collect data from.
            extractor: Feature extractor to use.
            stat: Stat to compute residuals for.

        Returns:
            Tuple of (features list, residuals list).
        """
        from fantasy_baseball_manager.pipeline.types import PlayerRates

        features: list[np.ndarray] = []
        residuals: list[float] = []

        for target_year in years:
            # Get projections for target year
            projections = self.pipeline.project_batters(self.data_source, target_year)
            proj_lookup = {p.player_id: p for p in projections}

            # Get actuals for target year
            actuals = self.data_source.batting_stats(target_year)
            actual_lookup = {a.player_id: a for a in actuals}

            # Get Statcast data from prior year
            statcast_year = target_year - 1
            statcast_data = self.statcast_source.batter_expected_stats(statcast_year)
            statcast_lookup = {s.player_id: s for s in statcast_data}

            # Get skill data from prior year
            skill_data = self.skill_data_source.batter_skill_stats(statcast_year)
            skill_lookup = {s.player_id: s for s in skill_data}

            for fg_id, proj in proj_lookup.items():
                actual = actual_lookup.get(fg_id)
                if actual is None or actual.pa < self.config.batter_min_pa:
                    continue

                mlbam_id = self.id_mapper.fangraphs_to_mlbam(fg_id)
                if mlbam_id is None:
                    continue

                statcast = statcast_lookup.get(mlbam_id)
                if statcast is None:
                    continue

                player_skill_data = skill_lookup.get(fg_id)

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

                feature_vec = extractor.extract(player_rates, statcast, player_skill_data)
                if feature_vec is None:
                    continue

                projected_count = getattr(proj, stat, 0)
                actual_count = getattr(actual, stat, 0)
                residual = actual_count - projected_count

                features.append(feature_vec)
                residuals.append(residual)

        return features, residuals

    def _collect_pitcher_data(
        self,
        years: tuple[int, ...],
        extractor: PitcherFeatureExtractor,
        stat: str,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Collect features and residuals for specified years.

        Args:
            years: Target years to collect data from.
            extractor: Feature extractor to use.
            stat: Stat to compute residuals for.

        Returns:
            Tuple of (features list, residuals list).
        """
        from fantasy_baseball_manager.pipeline.types import PlayerRates

        features: list[np.ndarray] = []
        residuals: list[float] = []

        for target_year in years:
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

            statcast_lookup = {s.player_id: s for s in statcast_data}
            bb_lookup = {b.player_id: b for b in batted_ball_data}

            for fg_id, proj in proj_lookup.items():
                actual = actual_lookup.get(fg_id)
                if actual is None:
                    continue

                actual_outs = int(actual.ip * 3)
                if actual_outs < self.config.pitcher_min_pa:
                    continue

                mlbam_id = self.id_mapper.fangraphs_to_mlbam(fg_id)
                if mlbam_id is None:
                    continue

                statcast = statcast_lookup.get(mlbam_id)
                if statcast is None:
                    continue

                batted_ball = bb_lookup.get(fg_id)

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

                feature_vec = extractor.extract(player_rates, statcast, batted_ball)
                if feature_vec is None:
                    continue

                projected_count = getattr(proj, stat, 0)
                actual_count = getattr(actual, stat, 0)
                residual = actual_count - projected_count

                features.append(feature_vec)
                residuals.append(residual)

        return features, residuals
