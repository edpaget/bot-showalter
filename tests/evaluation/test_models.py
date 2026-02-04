import pytest

from fantasy_baseball_manager.evaluation.models import (
    EvaluationResult,
    HeadToHeadResult,
    PlayerResidual,
    RankAccuracy,
    SourceEvaluation,
    StatAccuracy,
    StratumAccuracy,
)
from fantasy_baseball_manager.valuation.models import StatCategory


class TestStatAccuracy:
    def test_construction(self) -> None:
        sa = StatAccuracy(category=StatCategory.HR, sample_size=100, rmse=2.5, mae=1.8, correlation=0.95)
        assert sa.category == StatCategory.HR
        assert sa.sample_size == 100
        assert sa.rmse == 2.5
        assert sa.mae == 1.8
        assert sa.correlation == 0.95

    def test_frozen(self) -> None:
        sa = StatAccuracy(category=StatCategory.HR, sample_size=100, rmse=2.5, mae=1.8, correlation=0.95)
        with pytest.raises(AttributeError):
            sa.rmse = 3.0  # type: ignore[misc]


class TestRankAccuracy:
    def test_construction(self) -> None:
        ra = RankAccuracy(sample_size=50, spearman_rho=0.85, top_n=20, top_n_precision=0.7)
        assert ra.sample_size == 50
        assert ra.spearman_rho == 0.85
        assert ra.top_n == 20
        assert ra.top_n_precision == 0.7


class TestSourceEvaluation:
    def test_construction_with_none_rank_accuracy(self) -> None:
        se = SourceEvaluation(
            source_name="marcel",
            year=2024,
            batting_stat_accuracy=(),
            pitching_stat_accuracy=(),
            batting_rank_accuracy=None,
            pitching_rank_accuracy=None,
        )
        assert se.source_name == "marcel"
        assert se.year == 2024
        assert se.batting_rank_accuracy is None
        assert se.pitching_rank_accuracy is None

    def test_construction_with_rank_accuracy(self) -> None:
        ra = RankAccuracy(sample_size=50, spearman_rho=0.85, top_n=20, top_n_precision=0.7)
        sa = StatAccuracy(category=StatCategory.HR, sample_size=100, rmse=2.5, mae=1.8, correlation=0.95)
        se = SourceEvaluation(
            source_name="marcel",
            year=2024,
            batting_stat_accuracy=(sa,),
            pitching_stat_accuracy=(),
            batting_rank_accuracy=ra,
            pitching_rank_accuracy=None,
        )
        assert se.batting_stat_accuracy == (sa,)
        assert se.batting_rank_accuracy is ra


class TestEvaluationResult:
    def test_construction(self) -> None:
        se = SourceEvaluation(
            source_name="marcel",
            year=2024,
            batting_stat_accuracy=(),
            pitching_stat_accuracy=(),
            batting_rank_accuracy=None,
            pitching_rank_accuracy=None,
        )
        er = EvaluationResult(evaluations=(se,))
        assert len(er.evaluations) == 1
        assert er.evaluations[0].source_name == "marcel"


class TestStratumAccuracy:
    def test_construction(self) -> None:
        sa = StatAccuracy(category=StatCategory.HR, sample_size=50, rmse=2.5, mae=1.8, correlation=0.85)
        s = StratumAccuracy(stratum_name="PA 200-400", sample_size=50, stat_accuracy=(sa,), rank_accuracy=None)
        assert s.stratum_name == "PA 200-400"
        assert s.sample_size == 50
        assert len(s.stat_accuracy) == 1
        assert s.rank_accuracy is None

    def test_frozen(self) -> None:
        s = StratumAccuracy(stratum_name="PA 200-400", sample_size=50, stat_accuracy=(), rank_accuracy=None)
        with pytest.raises(AttributeError):
            s.sample_size = 100  # type: ignore[misc]


class TestPlayerResidual:
    def test_construction(self) -> None:
        r = PlayerResidual(
            player_id="b1",
            player_name="Test Player",
            category=StatCategory.HR,
            projected=25.0,
            actual=28.0,
            residual=3.0,
            abs_residual=3.0,
        )
        assert r.player_id == "b1"
        assert r.player_name == "Test Player"
        assert r.category == StatCategory.HR
        assert r.projected == 25.0
        assert r.actual == 28.0
        assert r.residual == 3.0
        assert r.abs_residual == 3.0

    def test_frozen(self) -> None:
        r = PlayerResidual(
            player_id="b1",
            player_name="Test Player",
            category=StatCategory.HR,
            projected=25.0,
            actual=28.0,
            residual=3.0,
            abs_residual=3.0,
        )
        with pytest.raises(AttributeError):
            r.residual = 5.0  # type: ignore[misc]


class TestHeadToHeadResult:
    def test_construction(self) -> None:
        h = HeadToHeadResult(
            source_a="marcel",
            source_b="marcel_gb",
            category=StatCategory.HR,
            sample_size=100,
            a_wins=50,
            b_wins=45,
            ties=5,
            a_win_pct=0.5,
            mean_improvement=1.2,
        )
        assert h.source_a == "marcel"
        assert h.source_b == "marcel_gb"
        assert h.category == StatCategory.HR
        assert h.sample_size == 100
        assert h.a_wins == 50
        assert h.b_wins == 45
        assert h.ties == 5
        assert h.a_win_pct == 0.5
        assert h.mean_improvement == 1.2

    def test_frozen(self) -> None:
        h = HeadToHeadResult(
            source_a="marcel",
            source_b="marcel_gb",
            category=StatCategory.HR,
            sample_size=100,
            a_wins=50,
            b_wins=45,
            ties=5,
            a_win_pct=0.5,
            mean_improvement=1.2,
        )
        with pytest.raises(AttributeError):
            h.a_wins = 60  # type: ignore[misc]


class TestSourceEvaluationBackwardsCompatible:
    def test_default_strata_fields(self) -> None:
        """Existing code without strata fields still works."""
        se = SourceEvaluation(
            source_name="test",
            year=2024,
            batting_stat_accuracy=(),
            pitching_stat_accuracy=(),
            batting_rank_accuracy=None,
            pitching_rank_accuracy=None,
        )
        assert se.batting_strata == ()
        assert se.pitching_strata == ()
        assert se.batting_residuals is None
        assert se.pitching_residuals is None
