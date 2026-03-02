import pytest

from fantasy_baseball_manager.domain.correlation_result import (
    CorrelationScanResult,
    MultiColumnRanking,
    PooledCorrelationResult,
    SeasonCorrelationResult,
    TargetCorrelation,
)


class TestTargetCorrelation:
    def test_construction(self) -> None:
        tc = TargetCorrelation(
            target="avg",
            pearson_r=0.85,
            pearson_p=0.001,
            spearman_rho=0.82,
            spearman_p=0.002,
            n=50,
        )
        assert tc.target == "avg"
        assert tc.pearson_r == 0.85
        assert tc.pearson_p == 0.001
        assert tc.spearman_rho == 0.82
        assert tc.spearman_p == 0.002
        assert tc.n == 50

    def test_frozen(self) -> None:
        tc = TargetCorrelation(target="avg", pearson_r=0.5, pearson_p=0.05, spearman_rho=0.4, spearman_p=0.06, n=10)
        with pytest.raises(AttributeError):
            tc.target = "obp"  # type: ignore[misc]


class TestSeasonCorrelationResult:
    def test_construction(self) -> None:
        corr = TargetCorrelation(target="slg", pearson_r=0.7, pearson_p=0.01, spearman_rho=0.65, spearman_p=0.02, n=30)
        result = SeasonCorrelationResult(
            column_spec="launch_speed",
            season=2023,
            player_type="batter",
            correlations=(corr,),
        )
        assert result.column_spec == "launch_speed"
        assert result.season == 2023
        assert result.player_type == "batter"
        assert len(result.correlations) == 1
        assert result.correlations[0].target == "slg"

    def test_frozen(self) -> None:
        result = SeasonCorrelationResult(
            column_spec="launch_speed",
            season=2023,
            player_type="batter",
            correlations=(),
        )
        with pytest.raises(AttributeError):
            result.season = 2024  # type: ignore[misc]


class TestPooledCorrelationResult:
    def test_construction(self) -> None:
        result = PooledCorrelationResult(
            column_spec="launch_speed",
            player_type="batter",
            correlations=(),
        )
        assert result.column_spec == "launch_speed"
        assert result.player_type == "batter"
        assert result.correlations == ()


class TestCorrelationScanResult:
    def test_construction(self) -> None:
        pooled = PooledCorrelationResult(
            column_spec="launch_speed",
            player_type="batter",
            correlations=(),
        )
        result = CorrelationScanResult(
            column_spec="launch_speed",
            player_type="batter",
            per_season=(),
            pooled=pooled,
        )
        assert result.column_spec == "launch_speed"
        assert result.player_type == "batter"
        assert result.per_season == ()
        assert result.pooled is pooled

    def test_frozen(self) -> None:
        pooled = PooledCorrelationResult(column_spec="x", player_type="batter", correlations=())
        result = CorrelationScanResult(column_spec="x", player_type="batter", per_season=(), pooled=pooled)
        with pytest.raises(AttributeError):
            result.column_spec = "y"  # type: ignore[misc]


class TestMultiColumnRanking:
    def test_construction(self) -> None:
        ranking = MultiColumnRanking(
            column_spec="launch_speed",
            avg_abs_pearson=0.45,
            avg_abs_spearman=0.42,
        )
        assert ranking.column_spec == "launch_speed"
        assert ranking.avg_abs_pearson == 0.45
        assert ranking.avg_abs_spearman == 0.42

    def test_frozen(self) -> None:
        ranking = MultiColumnRanking(column_spec="launch_speed", avg_abs_pearson=0.5, avg_abs_spearman=0.4)
        with pytest.raises(AttributeError):
            ranking.avg_abs_pearson = 0.9  # type: ignore[misc]
