"""Tests for validation framework."""

import numpy as np
import pytest

from fantasy_baseball_manager.ml.validation import (
    BaselineComparison,
    EarlyStoppingConfig,
    HoldoutEvaluation,
    HoldoutEvaluationReport,
    HoldoutEvaluator,
    LeaveOneYearOut,
    StatValidationResult,
    TimeSeriesHoldout,
    ValidationMetrics,
    ValidationReport,
    ValidationStrategy,
    compute_spearman_rho,
    compute_validation_metrics,
    compute_weighted_rmse,
)


class TestTimeSeriesHoldout:
    def test_generates_single_split(self) -> None:
        strategy = TimeSeriesHoldout(holdout_years=1)
        splits = strategy.generate_splits((2019, 2020, 2021, 2022))

        assert len(splits) == 1
        assert splits[0].fold_name == "holdout"
        assert splits[0].train_years == (2019, 2020, 2021)
        assert splits[0].val_years == (2022,)

    def test_multiple_holdout_years(self) -> None:
        strategy = TimeSeriesHoldout(holdout_years=2)
        splits = strategy.generate_splits((2019, 2020, 2021, 2022))

        assert len(splits) == 1
        assert splits[0].train_years == (2019, 2020)
        assert splits[0].val_years == (2021, 2022)

    def test_sorts_unsorted_years(self) -> None:
        strategy = TimeSeriesHoldout(holdout_years=1)
        splits = strategy.generate_splits((2022, 2019, 2021, 2020))

        assert splits[0].train_years == (2019, 2020, 2021)
        assert splits[0].val_years == (2022,)

    def test_raises_on_insufficient_years(self) -> None:
        strategy = TimeSeriesHoldout(holdout_years=2)

        with pytest.raises(ValueError, match="Need more than 2 years"):
            strategy.generate_splits((2021, 2022))

    def test_raises_on_equal_years_and_holdout(self) -> None:
        strategy = TimeSeriesHoldout(holdout_years=3)

        with pytest.raises(ValueError, match="Need more than 3 years"):
            strategy.generate_splits((2020, 2021, 2022))

    def test_name_property(self) -> None:
        assert TimeSeriesHoldout(holdout_years=1).name == "time_series_holdout_1y"
        assert TimeSeriesHoldout(holdout_years=2).name == "time_series_holdout_2y"

    def test_implements_protocol(self) -> None:
        strategy = TimeSeriesHoldout()
        assert isinstance(strategy, ValidationStrategy)


class TestLeaveOneYearOut:
    def test_generates_n_folds_for_n_years(self) -> None:
        strategy = LeaveOneYearOut()
        splits = strategy.generate_splits((2019, 2020, 2021))

        assert len(splits) == 3

    def test_each_fold_holds_out_one_year(self) -> None:
        strategy = LeaveOneYearOut()
        splits = strategy.generate_splits((2019, 2020, 2021))

        # First fold holds out 2019
        assert splits[0].train_years == (2020, 2021)
        assert splits[0].val_years == (2019,)
        assert splits[0].fold_name == "fold_1_holdout_2019"

        # Second fold holds out 2020
        assert splits[1].train_years == (2019, 2021)
        assert splits[1].val_years == (2020,)
        assert splits[1].fold_name == "fold_2_holdout_2020"

        # Third fold holds out 2021
        assert splits[2].train_years == (2019, 2020)
        assert splits[2].val_years == (2021,)
        assert splits[2].fold_name == "fold_3_holdout_2021"

    def test_sorts_unsorted_years(self) -> None:
        strategy = LeaveOneYearOut()
        splits = strategy.generate_splits((2022, 2020, 2021))

        # First holdout should be earliest year
        assert splits[0].val_years == (2020,)
        assert splits[1].val_years == (2021,)
        assert splits[2].val_years == (2022,)

    def test_raises_on_single_year(self) -> None:
        strategy = LeaveOneYearOut()

        with pytest.raises(ValueError, match="Need at least 2 years"):
            strategy.generate_splits((2022,))

    def test_name_property(self) -> None:
        assert LeaveOneYearOut().name == "leave_one_year_out"

    def test_implements_protocol(self) -> None:
        strategy = LeaveOneYearOut()
        assert isinstance(strategy, ValidationStrategy)


class TestComputeValidationMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_validation_metrics(y_true, y_pred, "test_fold")

        assert metrics.fold_name == "test_fold"
        assert metrics.n_samples == 5
        assert metrics.rmse == pytest.approx(0.0)
        assert metrics.mae == pytest.approx(0.0)
        assert metrics.r_squared == pytest.approx(1.0)

    def test_calculates_rmse_correctly(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # Errors: 1, 0, -1
        # MSE = (1 + 0 + 1) / 3 = 2/3
        # RMSE = sqrt(2/3) = 0.8165...

        metrics = compute_validation_metrics(y_true, y_pred, "test")

        expected_rmse = np.sqrt(2 / 3)
        assert metrics.rmse == pytest.approx(expected_rmse)

    def test_calculates_mae_correctly(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # Absolute errors: 1, 0, 1
        # MAE = (1 + 0 + 1) / 3 = 2/3

        metrics = compute_validation_metrics(y_true, y_pred, "test")

        expected_mae = 2 / 3
        assert metrics.mae == pytest.approx(expected_mae)

    def test_calculates_r_squared_correctly(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        metrics = compute_validation_metrics(y_true, y_pred, "test")

        # R² should be high for these good predictions
        assert metrics.r_squared > 0.95

    def test_r_squared_zero_for_constant_targets(self) -> None:
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([4.0, 5.0, 6.0])

        metrics = compute_validation_metrics(y_true, y_pred, "test")

        # When all targets are the same, R² should be 0
        assert metrics.r_squared == pytest.approx(0.0)

    def test_raises_on_empty_arrays(self) -> None:
        with pytest.raises(ValueError, match="empty arrays"):
            compute_validation_metrics(np.array([]), np.array([]), "test")

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_validation_metrics(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
                "test",
            )


class TestValidationMetrics:
    def test_frozen_dataclass(self) -> None:
        metrics = ValidationMetrics(
            fold_name="test",
            n_samples=100,
            rmse=1.5,
            mae=1.2,
            r_squared=0.85,
        )

        with pytest.raises(AttributeError):
            metrics.rmse = 2.0  # type: ignore[misc]


class TestStatValidationResult:
    def test_mean_rmse(self) -> None:
        result = StatValidationResult(
            stat_name="hr",
            fold_metrics=(
                ValidationMetrics("fold1", 100, rmse=1.0, mae=0.8, r_squared=0.9),
                ValidationMetrics("fold2", 100, rmse=2.0, mae=1.6, r_squared=0.8),
                ValidationMetrics("fold3", 100, rmse=1.5, mae=1.2, r_squared=0.85),
            ),
        )

        assert result.mean_rmse == pytest.approx(1.5)

    def test_mean_mae(self) -> None:
        result = StatValidationResult(
            stat_name="hr",
            fold_metrics=(
                ValidationMetrics("fold1", 100, rmse=1.0, mae=0.8, r_squared=0.9),
                ValidationMetrics("fold2", 100, rmse=2.0, mae=1.6, r_squared=0.8),
            ),
        )

        assert result.mean_mae == pytest.approx(1.2)

    def test_mean_r_squared(self) -> None:
        result = StatValidationResult(
            stat_name="hr",
            fold_metrics=(
                ValidationMetrics("fold1", 100, rmse=1.0, mae=0.8, r_squared=0.9),
                ValidationMetrics("fold2", 100, rmse=2.0, mae=1.6, r_squared=0.8),
            ),
        )

        assert result.mean_r_squared == pytest.approx(0.85)

    def test_total_samples(self) -> None:
        result = StatValidationResult(
            stat_name="hr",
            fold_metrics=(
                ValidationMetrics("fold1", 100, rmse=1.0, mae=0.8, r_squared=0.9),
                ValidationMetrics("fold2", 150, rmse=2.0, mae=1.6, r_squared=0.8),
            ),
        )

        assert result.total_samples == 250

    def test_empty_fold_metrics(self) -> None:
        result = StatValidationResult(stat_name="hr", fold_metrics=())

        assert result.mean_rmse == 0.0
        assert result.mean_mae == 0.0
        assert result.mean_r_squared == 0.0
        assert result.total_samples == 0


class TestValidationReport:
    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        original = ValidationReport(
            player_type="batter",
            strategy_name="time_series_holdout_1y",
            stat_results=(
                StatValidationResult(
                    stat_name="hr",
                    fold_metrics=(
                        ValidationMetrics("holdout", 100, rmse=2.5, mae=2.0, r_squared=0.75),
                    ),
                ),
                StatValidationResult(
                    stat_name="so",
                    fold_metrics=(
                        ValidationMetrics("holdout", 100, rmse=5.0, mae=4.0, r_squared=0.80),
                    ),
                ),
            ),
            training_years=(2019, 2020, 2021),
            holdout_years=(2022,),
        )

        serialized = original.to_dict()
        restored = ValidationReport.from_dict(serialized)

        assert restored.player_type == original.player_type
        assert restored.strategy_name == original.strategy_name
        assert restored.training_years == original.training_years
        assert restored.holdout_years == original.holdout_years
        assert len(restored.stat_results) == len(original.stat_results)
        assert restored.stat_results[0].stat_name == "hr"
        assert restored.stat_results[0].fold_metrics[0].rmse == 2.5


class TestEarlyStoppingConfig:
    def test_default_values(self) -> None:
        config = EarlyStoppingConfig()

        assert config.enabled is True
        assert config.patience == 10
        assert config.eval_fraction == 0.1

    def test_custom_values(self) -> None:
        config = EarlyStoppingConfig(enabled=False, patience=20, eval_fraction=0.2)

        assert config.enabled is False
        assert config.patience == 20
        assert config.eval_fraction == 0.2

    def test_frozen_dataclass(self) -> None:
        config = EarlyStoppingConfig()

        with pytest.raises(AttributeError):
            config.patience = 20  # type: ignore[misc]


# =============================================================================
# Holdout Evaluation Tests
# =============================================================================


class TestComputeWeightedRmse:
    def test_uniform_weights_equals_regular_rmse(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        weights = np.ones(5)

        weighted = compute_weighted_rmse(y_true, y_pred, weights)
        regular_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        assert weighted == pytest.approx(regular_rmse)

    def test_higher_weight_increases_contribution(self) -> None:
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 2.0])  # Error of 1 for first, 0 for second

        # Equal weights
        equal_weights = np.array([1.0, 1.0])
        equal_rmse = compute_weighted_rmse(y_true, y_pred, equal_weights)

        # Weight first sample more (the one with error)
        high_first = np.array([10.0, 1.0])
        high_first_rmse = compute_weighted_rmse(y_true, y_pred, high_first)

        # Weight second sample more (the one without error)
        high_second = np.array([1.0, 10.0])
        high_second_rmse = compute_weighted_rmse(y_true, y_pred, high_second)

        assert high_first_rmse > equal_rmse
        assert high_second_rmse < equal_rmse

    def test_perfect_predictions(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        weights = np.array([100.0, 200.0, 300.0])

        result = compute_weighted_rmse(y_true, y_pred, weights)

        assert result == pytest.approx(0.0)

    def test_raises_on_empty_arrays(self) -> None:
        with pytest.raises(ValueError, match="empty arrays"):
            compute_weighted_rmse(np.array([]), np.array([]), np.array([]))

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_weighted_rmse(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 1.0]),
            )

    def test_zero_weights_returns_zero(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 6.0, 7.0])
        weights = np.array([0.0, 0.0, 0.0])

        result = compute_weighted_rmse(y_true, y_pred, weights)

        assert result == 0.0


class TestComputeSpearmanRho:
    def test_perfect_positive_correlation(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        result = compute_spearman_rho(y_true, y_pred)

        assert result == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([50.0, 40.0, 30.0, 20.0, 10.0])

        result = compute_spearman_rho(y_true, y_pred)

        assert result == pytest.approx(-1.0)

    def test_no_correlation(self) -> None:
        # Values with no clear monotonic relationship
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_pred = np.array([3.0, 1.0, 4.0, 6.0, 2.0, 5.0])

        result = compute_spearman_rho(y_true, y_pred)

        # Should be close to 0 (no correlation)
        assert -0.5 < result < 0.5

    def test_handles_constant_array(self) -> None:
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        result = compute_spearman_rho(y_true, y_pred)

        # Should return 0 for constant array (NaN handled)
        assert result == 0.0

    def test_single_sample_returns_zero(self) -> None:
        result = compute_spearman_rho(np.array([1.0]), np.array([2.0]))

        assert result == 0.0

    def test_raises_on_empty_arrays(self) -> None:
        with pytest.raises(ValueError, match="empty arrays"):
            compute_spearman_rho(np.array([]), np.array([]))

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            compute_spearman_rho(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


class TestHoldoutEvaluation:
    def test_frozen_dataclass(self) -> None:
        eval_result = HoldoutEvaluation(
            stat_name="hr",
            n_samples=100,
            rmse=0.01,
            weighted_rmse=0.012,
            mae=0.008,
            spearman_rho=0.85,
        )

        with pytest.raises(AttributeError):
            eval_result.rmse = 0.02  # type: ignore[misc]


class TestHoldoutEvaluationReport:
    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        original = HoldoutEvaluationReport(
            model_name="test_model",
            test_years=(2024,),
            n_samples=150,
            stat_evaluations=(
                HoldoutEvaluation("hr", 150, 0.01, 0.012, 0.008, 0.85),
                HoldoutEvaluation("so", 150, 0.05, 0.055, 0.04, 0.90),
            ),
            baseline_comparisons=(
                BaselineComparison("hr", 0.01, 0.015, 0.005, 33.3, 0.85, 0.70),
            ),
        )

        serialized = original.to_dict()
        restored = HoldoutEvaluationReport.from_dict(serialized)

        assert restored.model_name == original.model_name
        assert restored.test_years == original.test_years
        assert restored.n_samples == original.n_samples
        assert len(restored.stat_evaluations) == 2
        assert restored.stat_evaluations[0].stat_name == "hr"
        assert restored.stat_evaluations[0].rmse == 0.01
        assert restored.baseline_comparisons is not None
        assert len(restored.baseline_comparisons) == 1
        assert restored.baseline_comparisons[0].rmse_improvement == 0.005

    def test_to_dict_without_baseline_comparisons(self) -> None:
        report = HoldoutEvaluationReport(
            model_name="test_model",
            test_years=(2024,),
            n_samples=100,
            stat_evaluations=(
                HoldoutEvaluation("hr", 100, 0.01, 0.012, 0.008, 0.85),
            ),
        )

        serialized = report.to_dict()

        assert "baseline_comparisons" not in serialized
        restored = HoldoutEvaluationReport.from_dict(serialized)
        assert restored.baseline_comparisons is None


class TestHoldoutEvaluator:
    def test_evaluate_computes_all_metrics(self) -> None:
        evaluator = HoldoutEvaluator(model_name="test", test_years=(2024,))
        y_true = {
            "hr": np.array([0.03, 0.04, 0.02, 0.05, 0.03]),
            "so": np.array([0.20, 0.25, 0.18, 0.22, 0.21]),
        }
        y_pred = {
            "hr": np.array([0.035, 0.038, 0.022, 0.048, 0.032]),
            "so": np.array([0.21, 0.24, 0.19, 0.23, 0.20]),
        }
        weights = np.array([200, 300, 150, 250, 400])

        report = evaluator.evaluate(y_true, y_pred, weights)

        assert report.model_name == "test"
        assert report.test_years == (2024,)
        assert report.n_samples == 5
        assert len(report.stat_evaluations) == 2

        # Find HR evaluation
        hr_eval = next(e for e in report.stat_evaluations if e.stat_name == "hr")
        assert hr_eval.n_samples == 5
        assert hr_eval.rmse > 0
        assert hr_eval.weighted_rmse > 0
        assert hr_eval.mae > 0
        assert -1 <= hr_eval.spearman_rho <= 1

    def test_evaluate_without_weights(self) -> None:
        evaluator = HoldoutEvaluator(model_name="test", test_years=(2024,))
        y_true = {"hr": np.array([0.03, 0.04, 0.02])}
        y_pred = {"hr": np.array([0.035, 0.038, 0.022])}

        report = evaluator.evaluate(y_true, y_pred)

        assert report.n_samples == 3
        assert len(report.stat_evaluations) == 1

    def test_evaluate_raises_on_stat_mismatch(self) -> None:
        evaluator = HoldoutEvaluator(model_name="test", test_years=(2024,))
        y_true = {"hr": np.array([0.03, 0.04])}
        y_pred = {"so": np.array([0.20, 0.25])}

        with pytest.raises(ValueError, match="Stat names don't match"):
            evaluator.evaluate(y_true, y_pred)

    def test_evaluate_raises_on_empty_stats(self) -> None:
        evaluator = HoldoutEvaluator(model_name="test", test_years=(2024,))

        with pytest.raises(ValueError, match="No stats provided"):
            evaluator.evaluate({}, {})

    def test_evaluate_raises_on_empty_arrays(self) -> None:
        evaluator = HoldoutEvaluator(model_name="test", test_years=(2024,))
        y_true = {"hr": np.array([])}
        y_pred = {"hr": np.array([])}

        with pytest.raises(ValueError, match="empty arrays"):
            evaluator.evaluate(y_true, y_pred)

    def test_evaluate_raises_on_length_mismatch(self) -> None:
        evaluator = HoldoutEvaluator(model_name="test", test_years=(2024,))
        y_true = {
            "hr": np.array([0.03, 0.04, 0.02]),
            "so": np.array([0.20, 0.25]),  # Wrong length
        }
        y_pred = {
            "hr": np.array([0.035, 0.038, 0.022]),
            "so": np.array([0.21, 0.24]),
        }

        with pytest.raises(ValueError, match="Array length mismatch"):
            evaluator.evaluate(y_true, y_pred)

    def test_compare_to_baseline(self) -> None:
        evaluator = HoldoutEvaluator(model_name="mle_v1", test_years=(2024,))
        y_true = {
            "hr": np.array([0.03, 0.04, 0.02, 0.05, 0.03]),
        }
        # Good model predictions
        y_pred = {
            "hr": np.array([0.031, 0.039, 0.021, 0.049, 0.031]),
        }
        # Worse baseline predictions
        y_baseline = {
            "hr": np.array([0.035, 0.035, 0.025, 0.045, 0.035]),
        }

        report = evaluator.compare_to_baseline(
            y_true, y_pred, y_baseline, baseline_name="traditional_mle"
        )

        assert "vs traditional_mle" in report.model_name
        assert report.baseline_comparisons is not None
        assert len(report.baseline_comparisons) == 1

        comparison = report.baseline_comparisons[0]
        assert comparison.stat_name == "hr"
        assert comparison.model_rmse < comparison.baseline_rmse
        assert comparison.rmse_improvement > 0
        assert comparison.rmse_improvement_pct > 0

    def test_compare_to_baseline_raises_on_stat_mismatch(self) -> None:
        evaluator = HoldoutEvaluator(model_name="test", test_years=(2024,))
        y_true = {"hr": np.array([0.03, 0.04])}
        y_pred = {"hr": np.array([0.035, 0.038])}
        y_baseline = {"so": np.array([0.20, 0.25])}

        with pytest.raises(ValueError, match="Baseline stat names don't match"):
            evaluator.compare_to_baseline(y_true, y_pred, y_baseline)
