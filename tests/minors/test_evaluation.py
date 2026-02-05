"""Tests for MLE evaluation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fantasy_baseball_manager.minors.evaluation import (
    MLEEvaluator,
    NoTranslationBaseline,
    TraditionalMLEBaseline,
    print_evaluation_report,
)
from fantasy_baseball_manager.minors.training_data import AggregatedMiLBStats
from fantasy_baseball_manager.minors.types import MinorLeagueLevel


def _make_aggregated_stats(
    player_id: str = "123",
    level: MinorLeagueLevel = MinorLeagueLevel.AAA,
    hr_rate: float = 0.03,
    so_rate: float = 0.20,
    bb_rate: float = 0.10,
    singles_rate: float = 0.15,
    doubles_rate: float = 0.05,
    triples_rate: float = 0.005,
    sb_rate: float = 0.10,
) -> AggregatedMiLBStats:
    """Create a mock AggregatedMiLBStats for testing."""
    return AggregatedMiLBStats(
        player_id=player_id,
        name=f"Player {player_id}",
        season=2023,
        age=24,
        total_pa=400,
        highest_level=level,
        pct_at_aaa=1.0 if level == MinorLeagueLevel.AAA else 0.0,
        pct_at_aa=1.0 if level == MinorLeagueLevel.AA else 0.0,
        pct_at_high_a=1.0 if level == MinorLeagueLevel.HIGH_A else 0.0,
        pct_at_single_a=1.0 if level == MinorLeagueLevel.SINGLE_A else 0.0,
        hr_rate=hr_rate,
        so_rate=so_rate,
        bb_rate=bb_rate,
        hit_rate=singles_rate + doubles_rate + triples_rate,
        singles_rate=singles_rate,
        doubles_rate=doubles_rate,
        triples_rate=triples_rate,
        sb_rate=sb_rate,
        iso=0.15,
        avg=0.250,
        obp=0.350,
        slg=0.400,
    )


class TestTraditionalMLEBaseline:
    def test_applies_aaa_factor(self) -> None:
        baseline = TraditionalMLEBaseline()
        stats = [_make_aggregated_stats(level=MinorLeagueLevel.AAA, hr_rate=0.04)]

        predictions = baseline.predict(stats)

        # AAA factor is 0.90
        assert predictions["hr"][0] == pytest.approx(0.04 * 0.90)

    def test_applies_aa_factor(self) -> None:
        baseline = TraditionalMLEBaseline()
        stats = [_make_aggregated_stats(level=MinorLeagueLevel.AA, hr_rate=0.04)]

        predictions = baseline.predict(stats)

        # AA factor is 0.80
        assert predictions["hr"][0] == pytest.approx(0.04 * 0.80)

    def test_applies_high_a_factor(self) -> None:
        baseline = TraditionalMLEBaseline()
        stats = [_make_aggregated_stats(level=MinorLeagueLevel.HIGH_A, hr_rate=0.04)]

        predictions = baseline.predict(stats)

        # High-A factor is 0.70
        assert predictions["hr"][0] == pytest.approx(0.04 * 0.70)

    def test_applies_single_a_factor(self) -> None:
        baseline = TraditionalMLEBaseline()
        stats = [_make_aggregated_stats(level=MinorLeagueLevel.SINGLE_A, hr_rate=0.04)]

        predictions = baseline.predict(stats)

        # Single-A factor is 0.60
        assert predictions["hr"][0] == pytest.approx(0.04 * 0.60)

    def test_predicts_all_stats(self) -> None:
        baseline = TraditionalMLEBaseline()
        stats = [
            _make_aggregated_stats(
                level=MinorLeagueLevel.AAA,
                hr_rate=0.03,
                so_rate=0.20,
                bb_rate=0.10,
                singles_rate=0.15,
                doubles_rate=0.05,
                triples_rate=0.005,
                sb_rate=0.10,
            )
        ]

        predictions = baseline.predict(stats)

        factor = 0.90  # AAA
        assert "hr" in predictions
        assert "so" in predictions
        assert "bb" in predictions
        assert "singles" in predictions
        assert "doubles" in predictions
        assert "triples" in predictions
        assert "sb" in predictions

        assert predictions["hr"][0] == pytest.approx(0.03 * factor)
        assert predictions["so"][0] == pytest.approx(0.20 * factor)
        assert predictions["bb"][0] == pytest.approx(0.10 * factor)

    def test_handles_multiple_players(self) -> None:
        baseline = TraditionalMLEBaseline()
        stats = [
            _make_aggregated_stats(player_id="1", level=MinorLeagueLevel.AAA, hr_rate=0.04),
            _make_aggregated_stats(player_id="2", level=MinorLeagueLevel.AA, hr_rate=0.05),
            _make_aggregated_stats(player_id="3", level=MinorLeagueLevel.HIGH_A, hr_rate=0.06),
        ]

        predictions = baseline.predict(stats)

        assert len(predictions["hr"]) == 3
        assert predictions["hr"][0] == pytest.approx(0.04 * 0.90)  # AAA
        assert predictions["hr"][1] == pytest.approx(0.05 * 0.80)  # AA
        assert predictions["hr"][2] == pytest.approx(0.06 * 0.70)  # High-A

    def test_custom_factors(self) -> None:
        custom_factors = {
            MinorLeagueLevel.AAA: 0.95,
            MinorLeagueLevel.AA: 0.85,
        }
        baseline = TraditionalMLEBaseline(factors=custom_factors)
        stats = [_make_aggregated_stats(level=MinorLeagueLevel.AAA, hr_rate=0.04)]

        predictions = baseline.predict(stats)

        assert predictions["hr"][0] == pytest.approx(0.04 * 0.95)


class TestNoTranslationBaseline:
    def test_returns_raw_rates(self) -> None:
        baseline = NoTranslationBaseline()
        stats = [
            _make_aggregated_stats(
                hr_rate=0.03,
                so_rate=0.20,
                bb_rate=0.10,
            )
        ]

        predictions = baseline.predict(stats)

        # No translation - should return raw rates
        assert predictions["hr"][0] == pytest.approx(0.03)
        assert predictions["so"][0] == pytest.approx(0.20)
        assert predictions["bb"][0] == pytest.approx(0.10)

    def test_level_does_not_affect_prediction(self) -> None:
        baseline = NoTranslationBaseline()
        aaa_stats = [_make_aggregated_stats(level=MinorLeagueLevel.AAA, hr_rate=0.04)]
        aa_stats = [_make_aggregated_stats(level=MinorLeagueLevel.AA, hr_rate=0.04)]

        aaa_predictions = baseline.predict(aaa_stats)
        aa_predictions = baseline.predict(aa_stats)

        # Same rate regardless of level
        assert aaa_predictions["hr"][0] == aa_predictions["hr"][0]


class TestMLEEvaluator:
    def test_evaluate_returns_report(self) -> None:
        """Test that evaluate returns a proper HoldoutEvaluationReport."""
        # Create mock data sources
        milb_source = MagicMock()
        mlb_source = MagicMock()

        # Create mock model
        model = MagicMock()
        model.predict_rates_batch.return_value = {
            "hr": np.array([0.03, 0.035]),
            "so": np.array([0.20, 0.22]),
            "bb": np.array([0.10, 0.11]),
            "singles": np.array([0.15, 0.16]),
            "doubles": np.array([0.05, 0.055]),
            "triples": np.array([0.005, 0.006]),
            "sb": np.array([0.10, 0.09]),
        }

        evaluator = MLEEvaluator(
            milb_source=milb_source,
            mlb_source=mlb_source,
        )

        # Mock the collector to return test data
        mock_features = np.array([[1, 2, 3], [4, 5, 6]])
        mock_targets = {
            "hr": np.array([0.032, 0.038]),
            "so": np.array([0.21, 0.23]),
            "bb": np.array([0.11, 0.12]),
            "singles": np.array([0.14, 0.15]),
            "doubles": np.array([0.048, 0.052]),
            "triples": np.array([0.004, 0.005]),
            "sb": np.array([0.11, 0.10]),
        }
        mock_weights = np.array([200, 300])
        mock_feature_names = ["f1", "f2", "f3"]
        mock_aggregated = [
            _make_aggregated_stats(player_id="1"),
            _make_aggregated_stats(player_id="2"),
        ]

        evaluator._cached_test_data[(2024,)] = (
            mock_features,
            mock_targets,
            mock_weights,
            mock_feature_names,
            mock_aggregated,
        )

        report = evaluator.evaluate(model, test_years=(2024,))

        assert report.model_name == "mle"
        assert report.test_years == (2024,)
        assert report.n_samples == 2
        assert len(report.stat_evaluations) == 7  # All 7 target stats

    def test_evaluate_with_empty_data(self) -> None:
        """Test evaluate handles empty test data gracefully."""
        milb_source = MagicMock()
        mlb_source = MagicMock()
        model = MagicMock()

        evaluator = MLEEvaluator(
            milb_source=milb_source,
            mlb_source=mlb_source,
        )

        # Mock empty data
        evaluator._cached_test_data[(2024,)] = (
            np.array([]).reshape(0, 3),
            {
                "hr": np.array([]),
                "so": np.array([]),
                "bb": np.array([]),
                "singles": np.array([]),
                "doubles": np.array([]),
                "triples": np.array([]),
                "sb": np.array([]),
            },
            np.array([]),
            ["f1", "f2", "f3"],
            [],
        )

        report = evaluator.evaluate(model, test_years=(2024,))

        assert report.n_samples == 0
        assert len(report.stat_evaluations) == 0

    def test_evaluate_with_baselines(self) -> None:
        """Test evaluate_with_baselines returns both model and baseline reports."""
        milb_source = MagicMock()
        mlb_source = MagicMock()

        model = MagicMock()
        model.predict_rates_batch.return_value = {
            "hr": np.array([0.03, 0.035]),
            "so": np.array([0.20, 0.22]),
            "bb": np.array([0.10, 0.11]),
            "singles": np.array([0.15, 0.16]),
            "doubles": np.array([0.05, 0.055]),
            "triples": np.array([0.005, 0.006]),
            "sb": np.array([0.10, 0.09]),
        }

        evaluator = MLEEvaluator(
            milb_source=milb_source,
            mlb_source=mlb_source,
        )

        mock_features = np.array([[1, 2, 3], [4, 5, 6]])
        mock_targets = {
            "hr": np.array([0.032, 0.038]),
            "so": np.array([0.21, 0.23]),
            "bb": np.array([0.11, 0.12]),
            "singles": np.array([0.14, 0.15]),
            "doubles": np.array([0.048, 0.052]),
            "triples": np.array([0.004, 0.005]),
            "sb": np.array([0.11, 0.10]),
        }
        mock_weights = np.array([200, 300])
        mock_feature_names = ["f1", "f2", "f3"]
        mock_aggregated = [
            _make_aggregated_stats(player_id="1", level=MinorLeagueLevel.AAA),
            _make_aggregated_stats(player_id="2", level=MinorLeagueLevel.AA),
        ]

        evaluator._cached_test_data[(2024,)] = (
            mock_features,
            mock_targets,
            mock_weights,
            mock_feature_names,
            mock_aggregated,
        )

        model_report, baseline_reports = evaluator.evaluate_with_baselines(model, test_years=(2024,))

        assert model_report.n_samples == 2
        assert "traditional_mle" in baseline_reports
        assert "no_translation" in baseline_reports

        # Traditional MLE report should have baseline comparisons
        trad_report = baseline_reports["traditional_mle"]
        assert trad_report.baseline_comparisons is not None
        assert len(trad_report.baseline_comparisons) == 7


class TestPrintEvaluationReport:
    def test_prints_without_error(self, capsys) -> None:
        """Test that print_evaluation_report runs without errors."""
        from fantasy_baseball_manager.ml.validation import (
            BaselineComparison,
            HoldoutEvaluation,
            HoldoutEvaluationReport,
        )

        report = HoldoutEvaluationReport(
            model_name="test_mle",
            test_years=(2024,),
            n_samples=100,
            stat_evaluations=(
                HoldoutEvaluation("hr", 100, 0.01, 0.012, 0.008, 0.85),
                HoldoutEvaluation("so", 100, 0.05, 0.055, 0.04, 0.90),
            ),
        )

        print_evaluation_report(report)

        captured = capsys.readouterr()
        assert "MLE Model Evaluation" in captured.out
        assert "test_mle" in captured.out
        assert "hr" in captured.out
        assert "so" in captured.out

    def test_prints_baseline_comparisons(self, capsys) -> None:
        """Test that baseline comparisons are printed."""
        from fantasy_baseball_manager.ml.validation import (
            BaselineComparison,
            HoldoutEvaluation,
            HoldoutEvaluationReport,
        )

        model_report = HoldoutEvaluationReport(
            model_name="mle",
            test_years=(2024,),
            n_samples=100,
            stat_evaluations=(HoldoutEvaluation("hr", 100, 0.01, 0.012, 0.008, 0.85),),
        )

        baseline_report = HoldoutEvaluationReport(
            model_name="mle vs traditional_mle",
            test_years=(2024,),
            n_samples=100,
            stat_evaluations=(HoldoutEvaluation("hr", 100, 0.01, 0.012, 0.008, 0.85),),
            baseline_comparisons=(BaselineComparison("hr", 0.01, 0.015, 0.005, 33.3, 0.85, 0.70),),
        )

        print_evaluation_report(model_report, {"traditional_mle": baseline_report})

        captured = capsys.readouterr()
        assert "Comparison vs traditional_mle" in captured.out
        assert "Improve" in captured.out
