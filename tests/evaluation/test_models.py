import pytest

from fantasy_baseball_manager.evaluation.models import (
    EvaluationResult,
    RankAccuracy,
    SourceEvaluation,
    StatAccuracy,
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
