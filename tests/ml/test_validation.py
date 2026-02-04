"""Tests for validation framework."""

import numpy as np
import pytest

from fantasy_baseball_manager.ml.validation import (
    EarlyStoppingConfig,
    LeaveOneYearOut,
    StatValidationResult,
    TimeSeriesHoldout,
    ValidationMetrics,
    ValidationReport,
    ValidationStrategy,
    compute_validation_metrics,
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
