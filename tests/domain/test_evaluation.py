import math

import pytest

from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    StatMetrics,
    SystemMetrics,
    compute_stat_metrics,
    summarize_comparison,
)
from fantasy_baseball_manager.domain.projection_accuracy import ProjectionComparison


def _metrics(rmse: float = 3.0, r_squared: float = 0.75) -> StatMetrics:
    return StatMetrics(rmse=rmse, mae=2.0, correlation=0.9, r_squared=r_squared, n=100)


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

    def test_labels_from_system_names(self) -> None:
        baseline = _system(system="steamer", version="2025")
        candidate = _system(system="zips", version="2025")
        result = _comparison(systems=[baseline, candidate])
        summary = summarize_comparison(result)

        assert summary.baseline_label == "steamer/2025"
        assert summary.candidate_label == "zips/2025"


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
