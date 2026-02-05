"""Evaluation framework for MLE models.

This module provides MLE-specific evaluation that builds on the generic
HoldoutEvaluator from ml.validation. It includes:
- MLEEvaluator for evaluating MLE models on held-out test data
- Traditional MLE baseline (fixed translation factors)
- No-translation baseline for comparison
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.minors.training_data import (
    BATTER_TARGET_STATS,
    AggregatedMiLBStats,
    MLETrainingDataCollector,
)
from fantasy_baseball_manager.minors.types import MinorLeagueLevel
from fantasy_baseball_manager.ml.validation import (
    HoldoutEvaluationReport,
    HoldoutEvaluator,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.minors.data_source import MinorLeagueDataSource
    from fantasy_baseball_manager.minors.model import MLEGradientBoostingModel
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


# Traditional MLE translation factors by level
# These are historical constants from baseball analysis literature
TRADITIONAL_MLE_FACTORS: dict[MinorLeagueLevel, float] = {
    MinorLeagueLevel.AAA: 0.90,
    MinorLeagueLevel.AA: 0.80,
    MinorLeagueLevel.HIGH_A: 0.70,
    MinorLeagueLevel.SINGLE_A: 0.60,
    MinorLeagueLevel.ROOKIE: 0.50,
}


@dataclass(frozen=True)
class TraditionalMLEBaseline:
    """Baseline that uses fixed translation factors per level.

    Traditional MLE applies a constant multiplier to MiLB rates
    based on the level of play (AAA=0.90, AA=0.80, etc.).
    """

    factors: dict[MinorLeagueLevel, float] = field(default_factory=lambda: TRADITIONAL_MLE_FACTORS.copy())

    def predict(
        self,
        aggregated_stats: list[AggregatedMiLBStats],
    ) -> dict[str, np.ndarray]:
        """Predict MLB rates using traditional fixed factors.

        Args:
            aggregated_stats: List of aggregated MiLB stats for each player

        Returns:
            Dict mapping stat name to array of predicted MLB rates
        """
        len(aggregated_stats)
        predictions: dict[str, list[float]] = {stat: [] for stat in BATTER_TARGET_STATS}

        for stats in aggregated_stats:
            factor = self.factors.get(stats.highest_level, 0.75)

            # Apply factor to each rate stat
            predictions["hr"].append(stats.hr_rate * factor)
            predictions["so"].append(stats.so_rate * factor)
            predictions["bb"].append(stats.bb_rate * factor)
            predictions["singles"].append(stats.singles_rate * factor)
            predictions["doubles"].append(stats.doubles_rate * factor)
            predictions["triples"].append(stats.triples_rate * factor)
            predictions["sb"].append(stats.sb_rate * factor)

        return {stat: np.array(vals) for stat, vals in predictions.items()}


@dataclass(frozen=True)
class NoTranslationBaseline:
    """Baseline that uses raw MiLB rates without any translation.

    This provides a lower bound - if the ML model can't beat raw MiLB rates,
    it's not learning useful translations.
    """

    def predict(
        self,
        aggregated_stats: list[AggregatedMiLBStats],
    ) -> dict[str, np.ndarray]:
        """Predict MLB rates as raw MiLB rates (no translation).

        Args:
            aggregated_stats: List of aggregated MiLB stats for each player

        Returns:
            Dict mapping stat name to array of predicted MLB rates
        """
        predictions: dict[str, list[float]] = {stat: [] for stat in BATTER_TARGET_STATS}

        for stats in aggregated_stats:
            predictions["hr"].append(stats.hr_rate)
            predictions["so"].append(stats.so_rate)
            predictions["bb"].append(stats.bb_rate)
            predictions["singles"].append(stats.singles_rate)
            predictions["doubles"].append(stats.doubles_rate)
            predictions["triples"].append(stats.triples_rate)
            predictions["sb"].append(stats.sb_rate)

        return {stat: np.array(vals) for stat, vals in predictions.items()}


@dataclass
class MLEEvaluator:
    """Evaluator for MLE models on held-out test data.

    This evaluator collects test data using MLETrainingDataCollector,
    generates predictions from the MLE model, and uses HoldoutEvaluator
    to compute metrics. It also supports comparison against baselines
    (traditional MLE, no translation).

    Example:
        evaluator = MLEEvaluator(
            milb_source=milb_data_source,
            mlb_source=mlb_data_source,
        )
        report = evaluator.evaluate(
            model=trained_mle_model,
            test_years=(2024,),
        )
        # Or with baseline comparison:
        report = evaluator.evaluate_with_baselines(
            model=trained_mle_model,
            test_years=(2024,),
        )
    """

    milb_source: MinorLeagueDataSource
    mlb_source: StatsDataSource
    min_milb_pa: int = 200
    min_mlb_pa: int = 100
    max_prior_mlb_pa: int = 200
    id_mapper: PlayerIdMapper | None = None

    # Cached collector and test data
    _collector: MLETrainingDataCollector | None = field(default=None, init=False, repr=False)
    _cached_test_data: dict[tuple[int, ...], tuple] = field(default_factory=dict, init=False, repr=False)

    def _get_collector(self) -> MLETrainingDataCollector:
        """Get or create the training data collector."""
        if self._collector is None:
            self._collector = MLETrainingDataCollector(
                milb_source=self.milb_source,
                mlb_source=self.mlb_source,
                min_milb_pa=self.min_milb_pa,
                min_mlb_pa=self.min_mlb_pa,
                max_prior_mlb_pa=self.max_prior_mlb_pa,
                id_mapper=self.id_mapper,
            )
        return self._collector

    def _collect_test_data(
        self,
        test_years: tuple[int, ...],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, list[str], list[AggregatedMiLBStats]]:
        """Collect test data for evaluation.

        Returns:
            Tuple of (features, targets, sample_weights, feature_names, aggregated_stats)
        """
        cache_key = test_years
        if cache_key in self._cached_test_data:
            return self._cached_test_data[cache_key]

        collector = self._get_collector()

        # Collect using the training data collector, including aggregated stats
        features, targets, weights, feature_names, aggregated_stats = collector.collect(
            test_years, include_aggregated_stats=True
        )

        # aggregated_stats is guaranteed to be a list when include_aggregated_stats=True
        aggregated_list = aggregated_stats if aggregated_stats is not None else []
        cached_result = (features, targets, weights, feature_names, aggregated_list)
        self._cached_test_data[cache_key] = cached_result
        return cached_result

    def evaluate(
        self,
        model: MLEGradientBoostingModel,
        test_years: tuple[int, ...],
    ) -> HoldoutEvaluationReport:
        """Evaluate MLE model on held-out test data.

        Args:
            model: Trained MLE gradient boosting model
            test_years: Years to use as test set (features from year-1, targets from year)

        Returns:
            HoldoutEvaluationReport with per-stat metrics
        """
        features, targets, weights, _feature_names, _ = self._collect_test_data(test_years)

        if len(features) == 0:
            logger.warning("No test samples found for years %s", test_years)
            return HoldoutEvaluationReport(
                model_name="mle",
                test_years=test_years,
                n_samples=0,
                stat_evaluations=(),
            )

        # Get model predictions
        predictions = model.predict_rates_batch(features)

        # Use generic holdout evaluator
        evaluator = HoldoutEvaluator(
            model_name="mle",
            test_years=test_years,
        )
        return evaluator.evaluate(targets, predictions, weights)

    def evaluate_with_baselines(
        self,
        model: MLEGradientBoostingModel,
        test_years: tuple[int, ...],
        include_traditional: bool = True,
        include_no_translation: bool = True,
    ) -> tuple[HoldoutEvaluationReport, dict[str, HoldoutEvaluationReport]]:
        """Evaluate MLE model and compare to baselines.

        Args:
            model: Trained MLE gradient boosting model
            test_years: Years to use as test set
            include_traditional: Include traditional MLE baseline comparison
            include_no_translation: Include no-translation baseline comparison

        Returns:
            Tuple of (model_report, baseline_reports_dict)
        """
        features, targets, weights, _feature_names, aggregated_stats = self._collect_test_data(test_years)

        if len(features) == 0:
            logger.warning("No test samples found for years %s", test_years)
            empty_report = HoldoutEvaluationReport(
                model_name="mle",
                test_years=test_years,
                n_samples=0,
                stat_evaluations=(),
            )
            return empty_report, {}

        # Get model predictions
        predictions = model.predict_rates_batch(features)

        # Create evaluator
        evaluator = HoldoutEvaluator(
            model_name="mle",
            test_years=test_years,
        )

        baseline_reports: dict[str, HoldoutEvaluationReport] = {}

        # Traditional MLE baseline
        if include_traditional:
            traditional_baseline = TraditionalMLEBaseline()
            traditional_predictions = traditional_baseline.predict(aggregated_stats)
            traditional_report = evaluator.compare_to_baseline(
                targets,
                predictions,
                traditional_predictions,
                weights,
                baseline_name="traditional_mle",
            )
            baseline_reports["traditional_mle"] = traditional_report

        # No translation baseline
        if include_no_translation:
            no_translation_baseline = NoTranslationBaseline()
            no_translation_predictions = no_translation_baseline.predict(aggregated_stats)
            no_translation_report = evaluator.compare_to_baseline(
                targets,
                predictions,
                no_translation_predictions,
                weights,
                baseline_name="no_translation",
            )
            baseline_reports["no_translation"] = no_translation_report

        # Get model-only report
        model_report = evaluator.evaluate(targets, predictions, weights)

        return model_report, baseline_reports


def print_evaluation_report(
    report: HoldoutEvaluationReport,
    baseline_reports: dict[str, HoldoutEvaluationReport] | None = None,
) -> None:
    """Print a human-readable evaluation report.

    Args:
        report: The main model evaluation report
        baseline_reports: Optional dict of baseline comparison reports
    """
    print(f"\n{'=' * 60}")
    print(f"MLE Model Evaluation: {report.model_name}")
    print(f"Test Years: {report.test_years}")
    print(f"Samples: {report.n_samples}")
    print(f"{'=' * 60}\n")

    # Print per-stat metrics
    print("Per-Stat Metrics:")
    print("-" * 60)
    print(f"{'Stat':<10} {'RMSE':>10} {'W-RMSE':>10} {'MAE':>10} {'Spearman':>10}")
    print("-" * 60)
    for stat_eval in report.stat_evaluations:
        print(
            f"{stat_eval.stat_name:<10} "
            f"{stat_eval.rmse:>10.5f} "
            f"{stat_eval.weighted_rmse:>10.5f} "
            f"{stat_eval.mae:>10.5f} "
            f"{stat_eval.spearman_rho:>10.3f}"
        )
    print()

    # Print baseline comparisons if available
    if baseline_reports:
        for baseline_name, baseline_report in baseline_reports.items():
            if baseline_report.baseline_comparisons:
                print(f"\nComparison vs {baseline_name}:")
                print("-" * 60)
                print(f"{'Stat':<10} {'Model':>10} {'Baseline':>10} {'Improve':>10} {'Pct':>8}")
                print("-" * 60)
                for comp in baseline_report.baseline_comparisons:
                    print(
                        f"{comp.stat_name:<10} "
                        f"{comp.model_rmse:>10.5f} "
                        f"{comp.baseline_rmse:>10.5f} "
                        f"{comp.rmse_improvement:>10.5f} "
                        f"{comp.rmse_improvement_pct:>7.1f}%"
                    )
