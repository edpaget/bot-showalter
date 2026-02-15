import math

import pytest

from fantasy_baseball_manager.domain.evaluation import compute_stat_metrics
from fantasy_baseball_manager.domain.projection_accuracy import ProjectionComparison


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
