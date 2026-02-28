import math

import pytest

from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    StatMetrics,
    SystemMetrics,
    check_regression,
    compute_stat_metrics,
    compute_tail_accuracy,
    summarize_comparison,
)
from fantasy_baseball_manager.domain.projection_accuracy import ProjectionComparison


def _metrics(rmse: float = 3.0, r_squared: float = 0.75, rank_correlation: float = 0.9) -> StatMetrics:
    return StatMetrics(
        rmse=rmse, mae=2.0, correlation=0.9, rank_correlation=rank_correlation, r_squared=r_squared, n=100
    )


def _system(
    system: str = "steamer",
    version: str = "2025",
    metrics: dict[str, StatMetrics] | None = None,
) -> SystemMetrics:
    if metrics is None:
        metrics = {"hr": _metrics(), "avg": _metrics()}
    return SystemMetrics(system=system, version=version, source_type="third_party", metrics=metrics)


def _comparison(
    stats: list[str] | None = None,
    systems: list[SystemMetrics] | None = None,
) -> ComparisonResult:
    if stats is None:
        stats = ["hr", "avg"]
    if systems is None:
        systems = [_system(), _system(system="zips")]
    return ComparisonResult(season=2024, stats=stats, systems=systems)


class TestSummarizeComparison:
    def test_candidate_wins_rmse(self) -> None:
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rmse=4.0),
                "avg": _metrics(rmse=3.5),
                "sb": _metrics(rmse=5.0),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rmse=3.0),
                "avg": _metrics(rmse=2.5),
                "sb": _metrics(rmse=6.0),
            },
        )
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)

        assert summary.rmse_wins == 2
        assert summary.rmse_losses == 1

    def test_baseline_wins_r_squared(self) -> None:
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(r_squared=0.80),
                "avg": _metrics(r_squared=0.70),
                "sb": _metrics(r_squared=0.60),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(r_squared=0.75),
                "avg": _metrics(r_squared=0.65),
                "sb": _metrics(r_squared=0.90),
            },
        )
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)

        assert summary.r_squared_wins == 1
        assert summary.r_squared_losses == 2

    def test_tie_on_identical_metrics(self) -> None:
        baseline = _system(system="steamer", metrics={"hr": _metrics(rmse=3.0, r_squared=0.75)})
        candidate = _system(system="zips", metrics={"hr": _metrics(rmse=3.0, r_squared=0.75)})
        result = _comparison(stats=["hr"], systems=[baseline, candidate])
        summary = summarize_comparison(result)

        assert summary.rmse_ties == 1
        assert summary.r_squared_ties == 1
        assert summary.rmse_wins == 0
        assert summary.rmse_losses == 0

    def test_pct_delta_computed_correctly(self) -> None:
        baseline = _system(system="steamer", metrics={"hr": _metrics(rmse=4.0, r_squared=0.50)})
        candidate = _system(system="zips", metrics={"hr": _metrics(rmse=3.0, r_squared=0.75)})
        result = _comparison(stats=["hr"], systems=[baseline, candidate])
        summary = summarize_comparison(result)

        rec = summary.records[0]
        # RMSE delta: 3.0 - 4.0 = -1.0, pct: -1.0/4.0 = -25%
        assert rec.rmse_delta == pytest.approx(-1.0)
        assert rec.rmse_pct_delta == pytest.approx(-25.0)
        # R² delta: 0.75 - 0.50 = 0.25, pct: 0.25/0.50 = 50%
        assert rec.r_squared_delta == pytest.approx(0.25)
        assert rec.r_squared_pct_delta == pytest.approx(50.0)

    def test_skips_missing_stats(self) -> None:
        baseline = _system(system="steamer", metrics={"hr": _metrics(), "sb": _metrics()})
        candidate = _system(system="zips", metrics={"hr": _metrics(), "avg": _metrics()})
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)

        stat_names = [r.stat_name for r in summary.records]
        assert "hr" in stat_names
        assert "avg" not in stat_names
        assert "sb" not in stat_names

    def test_rank_correlation_wins_tallied(self) -> None:
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rank_correlation=0.80),
                "avg": _metrics(rank_correlation=0.70),
                "sb": _metrics(rank_correlation=0.90),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rank_correlation=0.85),
                "avg": _metrics(rank_correlation=0.60),
                "sb": _metrics(rank_correlation=0.90),
            },
        )
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)

        assert summary.rank_correlation_wins == 1  # hr
        assert summary.rank_correlation_losses == 1  # avg
        assert summary.rank_correlation_ties == 1  # sb

    def test_summarize_includes_rank_correlation_delta(self) -> None:
        baseline = _system(system="steamer", metrics={"hr": _metrics(rank_correlation=0.80)})
        candidate = _system(system="zips", metrics={"hr": _metrics(rank_correlation=0.90)})
        result = _comparison(stats=["hr"], systems=[baseline, candidate])
        summary = summarize_comparison(result)

        rec = summary.records[0]
        assert rec.baseline_rank_correlation == 0.80
        assert rec.candidate_rank_correlation == 0.90
        assert rec.rank_correlation_delta == pytest.approx(0.10)
        assert rec.rank_correlation_pct_delta == pytest.approx(12.5)
        assert rec.rank_correlation_winner == "candidate"

    def test_labels_from_system_names(self) -> None:
        baseline = _system(system="steamer", version="2025")
        candidate = _system(system="zips", version="2025")
        result = _comparison(systems=[baseline, candidate])
        summary = summarize_comparison(result)

        assert summary.baseline_label == "steamer/2025"
        assert summary.candidate_label == "zips/2025"


class TestRankCorrelation:
    def test_rank_correlation_perfect_ranking(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=30.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=20.0, actual=20.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=10.0, actual=10.0, error=0.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].rank_correlation == pytest.approx(1.0)

    def test_rank_correlation_reversed(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=10.0, error=20.0),
            ProjectionComparison(stat_name="hr", projected=20.0, actual=20.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=10.0, actual=30.0, error=-20.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].rank_correlation == pytest.approx(-1.0)

    def test_rank_correlation_single_player_zero(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].rank_correlation == 0.0

    def test_rank_correlation_with_ties(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=30.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=30.0, actual=20.0, error=10.0),
            ProjectionComparison(stat_name="hr", projected=10.0, actual=10.0, error=0.0),
        ]
        result = compute_stat_metrics(comparisons)
        # Tied projected values get average ranks; correlation should still be positive
        assert result["hr"].rank_correlation > 0.0

    def test_rank_correlation_constant_input_returns_zero(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=20.0, actual=10.0, error=10.0),
            ProjectionComparison(stat_name="hr", projected=20.0, actual=20.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=20.0, actual=30.0, error=-10.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].rank_correlation == 0.0


class TestCheckRegression:
    def test_candidate_wins_majority_on_both_passes(self) -> None:
        """Candidate wins majority on both RMSE and ρ → passes."""
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rmse=4.0, rank_correlation=0.70),
                "avg": _metrics(rmse=3.5, rank_correlation=0.60),
                "sb": _metrics(rmse=5.0, rank_correlation=0.90),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rmse=3.0, rank_correlation=0.80),
                "avg": _metrics(rmse=2.5, rank_correlation=0.70),
                "sb": _metrics(rmse=6.0, rank_correlation=0.80),
            },
        )
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is True
        assert check.rmse_passed is True
        assert check.rank_correlation_passed is True
        assert "PASS" in check.explanation

    def test_candidate_loses_majority_rmse_fails(self) -> None:
        """Candidate loses majority on RMSE but wins ρ → fails (rmse_passed=False)."""
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rmse=3.0, rank_correlation=0.70),
                "avg": _metrics(rmse=2.5, rank_correlation=0.60),
                "sb": _metrics(rmse=5.0, rank_correlation=0.80),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rmse=4.0, rank_correlation=0.80),
                "avg": _metrics(rmse=3.5, rank_correlation=0.70),
                "sb": _metrics(rmse=4.0, rank_correlation=0.90),
            },
        )
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is False
        assert check.rmse_passed is False
        assert check.rank_correlation_passed is True
        assert "FAIL" in check.explanation

    def test_candidate_wins_rmse_loses_majority_rank_correlation_fails(self) -> None:
        """Candidate wins RMSE but loses majority on ρ → fails (rank_correlation_passed=False)."""
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rmse=4.0, rank_correlation=0.90),
                "avg": _metrics(rmse=3.5, rank_correlation=0.80),
                "sb": _metrics(rmse=5.0, rank_correlation=0.70),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rmse=3.0, rank_correlation=0.80),
                "avg": _metrics(rmse=2.5, rank_correlation=0.70),
                "sb": _metrics(rmse=4.0, rank_correlation=0.80),
            },
        )
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is False
        assert check.rmse_passed is True
        assert check.rank_correlation_passed is False
        assert "FAIL" in check.explanation

    def test_candidate_loses_majority_on_both_fails(self) -> None:
        """Candidate loses majority on both RMSE and ρ → fails."""
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rmse=3.0, rank_correlation=0.90),
                "avg": _metrics(rmse=2.5, rank_correlation=0.80),
                "sb": _metrics(rmse=5.0, rank_correlation=0.70),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rmse=4.0, rank_correlation=0.80),
                "avg": _metrics(rmse=3.5, rank_correlation=0.70),
                "sb": _metrics(rmse=4.0, rank_correlation=0.80),
            },
        )
        result = _comparison(stats=["hr", "avg", "sb"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is False
        assert check.rmse_passed is False
        assert check.rank_correlation_passed is False

    def test_even_split_passes(self) -> None:
        """Even split (no strict majority lost) → passes."""
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rmse=3.0, rank_correlation=0.80),
                "avg": _metrics(rmse=4.0, rank_correlation=0.70),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rmse=4.0, rank_correlation=0.70),
                "avg": _metrics(rmse=3.0, rank_correlation=0.80),
            },
        )
        result = _comparison(stats=["hr", "avg"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is True
        assert check.rmse_passed is True
        assert check.rank_correlation_passed is True

    def test_all_ties_passes(self) -> None:
        """All ties → passes."""
        baseline = _system(
            system="steamer",
            metrics={
                "hr": _metrics(rmse=3.0, rank_correlation=0.80),
                "avg": _metrics(rmse=4.0, rank_correlation=0.70),
            },
        )
        candidate = _system(
            system="zips",
            metrics={
                "hr": _metrics(rmse=3.0, rank_correlation=0.80),
                "avg": _metrics(rmse=4.0, rank_correlation=0.70),
            },
        )
        result = _comparison(stats=["hr", "avg"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is True
        assert check.rmse_passed is True
        assert check.rank_correlation_passed is True

    def test_single_stat_candidate_wins_passes(self) -> None:
        """Single stat, candidate wins → passes."""
        baseline = _system(system="steamer", metrics={"hr": _metrics(rmse=4.0, rank_correlation=0.70)})
        candidate = _system(system="zips", metrics={"hr": _metrics(rmse=3.0, rank_correlation=0.80)})
        result = _comparison(stats=["hr"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is True

    def test_single_stat_candidate_loses_fails(self) -> None:
        """Single stat, candidate loses → fails."""
        baseline = _system(system="steamer", metrics={"hr": _metrics(rmse=3.0, rank_correlation=0.80)})
        candidate = _system(system="zips", metrics={"hr": _metrics(rmse=4.0, rank_correlation=0.70)})
        result = _comparison(stats=["hr"], systems=[baseline, candidate])
        summary = summarize_comparison(result)
        check = check_regression(summary)

        assert check.passed is False


class TestComputeStatMetricsBasic:
    def test_three_players_known_errors(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=25.0, actual=20.0, error=5.0),
            ProjectionComparison(stat_name="hr", projected=15.0, actual=18.0, error=-3.0),
            ProjectionComparison(stat_name="avg", projected=0.280, actual=0.265, error=0.015),
            ProjectionComparison(stat_name="avg", projected=0.300, actual=0.310, error=-0.010),
            ProjectionComparison(stat_name="avg", projected=0.250, actual=0.240, error=0.010),
        ]
        result = compute_stat_metrics(comparisons)

        assert "hr" in result
        assert "avg" in result
        assert result["hr"].n == 3
        assert result["avg"].n == 3

        # HR: errors = [2, 5, -3], MAE = (2+5+3)/3 = 10/3
        assert result["hr"].mae == pytest.approx(10.0 / 3.0)
        # HR: RMSE = sqrt((4+25+9)/3) = sqrt(38/3)
        assert result["hr"].rmse == pytest.approx(math.sqrt(38.0 / 3.0))
        # HR: correlation between [30,25,15] and [28,20,18]
        assert result["hr"].correlation != 0.0

    def test_single_player_correlation_zero(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].n == 1
        assert result["hr"].correlation == 0.0
        assert result["hr"].rank_correlation == 0.0

    def test_perfect_predictions(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=30.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=25.0, actual=25.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=15.0, actual=15.0, error=0.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].rmse == 0.0
        assert result["hr"].mae == 0.0

    def test_r_squared_perfect_predictions(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=30.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=25.0, actual=25.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=15.0, actual=15.0, error=0.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].r_squared == pytest.approx(1.0)

    def test_r_squared_zero_when_predictions_equal_mean(self) -> None:
        # All predictions = mean of actuals (22.0) → SS_res == SS_tot → R² = 0
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=22.0, actual=30.0, error=-8.0),
            ProjectionComparison(stat_name="hr", projected=22.0, actual=25.0, error=-3.0),
            ProjectionComparison(stat_name="hr", projected=22.0, actual=11.0, error=11.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].r_squared == pytest.approx(0.0)

    def test_r_squared_negative_when_worse_than_mean(self) -> None:
        # Predictions that are further from actuals than the mean would be
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=10.0, actual=30.0, error=-20.0),
            ProjectionComparison(stat_name="hr", projected=40.0, actual=20.0, error=20.0),
            ProjectionComparison(stat_name="hr", projected=5.0, actual=25.0, error=-20.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].r_squared < 0.0

    def test_r_squared_known_values(self) -> None:
        # actuals = [2, 4, 6], mean = 4
        # projected = [3, 4, 5], errors = [1, 0, -1]
        # SS_res = 1+0+1 = 2, SS_tot = 4+0+4 = 8, R² = 1 - 2/8 = 0.75
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=3.0, actual=2.0, error=1.0),
            ProjectionComparison(stat_name="hr", projected=4.0, actual=4.0, error=0.0),
            ProjectionComparison(stat_name="hr", projected=5.0, actual=6.0, error=-1.0),
        ]
        result = compute_stat_metrics(comparisons)
        assert result["hr"].r_squared == pytest.approx(0.75)

    def test_filters_requested_stats(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
            ProjectionComparison(stat_name="avg", projected=0.280, actual=0.265, error=0.015),
            ProjectionComparison(stat_name="sb", projected=15.0, actual=12.0, error=3.0),
        ]
        result = compute_stat_metrics(comparisons, stats=["hr", "avg"])
        assert "hr" in result
        assert "avg" in result
        assert "sb" not in result


class TestComputeTailAccuracy:
    def test_tail_accuracy_top_n_subset(self) -> None:
        # 5 players with projected HR values; only top-3 should be used for n=3
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=40.0, actual=38.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=35.0, actual=30.0, error=5.0),
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=20.0, actual=22.0, error=-2.0),
            ProjectionComparison(stat_name="hr", projected=10.0, actual=15.0, error=-5.0),
        ]
        result = compute_tail_accuracy(comparisons, ns=(3,))
        assert "hr" in result.rmse_by_stat
        # Top 3 by projected: errors = [2, 5, 2], RMSE = sqrt((4+25+4)/3)
        expected = math.sqrt(33.0 / 3.0)
        assert result.rmse_by_stat["hr"][3] == pytest.approx(expected)

    def test_tail_accuracy_multiple_ns(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=float(50 - i), actual=float(48 - i), error=2.0)
            for i in range(10)
        ]
        result = compute_tail_accuracy(comparisons, ns=(3, 5))
        assert 3 in result.rmse_by_stat["hr"]
        assert 5 in result.rmse_by_stat["hr"]

    def test_tail_accuracy_filters_stats(self) -> None:
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=25.0, actual=23.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=20.0, actual=18.0, error=2.0),
            ProjectionComparison(stat_name="avg", projected=0.300, actual=0.290, error=0.010),
            ProjectionComparison(stat_name="avg", projected=0.280, actual=0.270, error=0.010),
            ProjectionComparison(stat_name="avg", projected=0.260, actual=0.250, error=0.010),
        ]
        result = compute_tail_accuracy(comparisons, ns=(2,), stats=["hr"])
        assert "hr" in result.rmse_by_stat
        assert "avg" not in result.rmse_by_stat

    def test_tail_accuracy_skips_small_stats(self) -> None:
        # Only 2 comparisons, but we ask for top-5
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=25.0, actual=23.0, error=2.0),
        ]
        result = compute_tail_accuracy(comparisons, ns=(5,))
        assert "hr" not in result.rmse_by_stat

    def test_tail_accuracy_partial_ns(self) -> None:
        # 3 comparisons: enough for n=3 but not for n=5
        comparisons = [
            ProjectionComparison(stat_name="hr", projected=30.0, actual=28.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=25.0, actual=23.0, error=2.0),
            ProjectionComparison(stat_name="hr", projected=20.0, actual=18.0, error=2.0),
        ]
        result = compute_tail_accuracy(comparisons, ns=(3, 5))
        assert 3 in result.rmse_by_stat["hr"]
        assert 5 not in result.rmse_by_stat["hr"]
