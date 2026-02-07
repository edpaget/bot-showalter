from collections.abc import Generator
from typing import Any

import pytest

from fantasy_baseball_manager.context import init_context, reset_context
from fantasy_baseball_manager.evaluation.harness import (
    EvaluationConfig,
    StratificationConfig,
    compare_sources,
    evaluate,
    evaluate_source,
)
from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.result import Ok
from fantasy_baseball_manager.valuation.models import StatCategory
from fantasy_baseball_manager.valuation.projection_source import SimpleProjectionSource


def _batter_proj(
    player_id: str = "b1",
    name: str = "Hitter",
    hr: float = 25.0,
    sb: float = 10.0,
    pa: float = 600.0,
    h: float = 160.0,
    bb: float = 50.0,
    hbp: float = 5.0,
    age: int = 28,
) -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=name,
        year=2024,
        age=age,
        pa=pa,
        ab=540.0,
        h=h,
        singles=h - 60,
        doubles=30.0,
        triples=5.0,
        hr=hr,
        bb=bb,
        so=120.0,
        hbp=hbp,
        sf=3.0,
        sh=2.0,
        sb=sb,
        cs=3.0,
        r=0.0,
        rbi=0.0,
    )


def _pitcher_proj(
    player_id: str = "p1",
    name: str = "Pitcher",
    so: float = 200.0,
    ip: float = 180.0,
    er: float = 60.0,
    h: float = 150.0,
    bb: float = 50.0,
) -> PitchingProjection:
    era = (er / ip * 9) if ip > 0 else 0.0
    whip = ((h + bb) / ip) if ip > 0 else 0.0
    return PitchingProjection(
        player_id=player_id,
        name=name,
        year=2024,
        age=27,
        ip=ip,
        g=32.0,
        gs=32.0,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=20.0,
        hbp=5.0,
        era=era,
        whip=whip,
        w=0.0,
        nsvh=0.0,
    )


def _batting_stats(
    player_id: str = "b1",
    name: str = "Hitter",
    pa: int = 600,
    hr: int = 25,
    sb: int = 10,
    age: int = 28,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=2024,
        age=age,
        pa=pa,
        ab=540,
        h=160,
        singles=100,
        doubles=30,
        triples=5,
        hr=hr,
        bb=50,
        so=120,
        hbp=5,
        sf=3,
        sh=2,
        sb=sb,
        cs=3,
        r=80,
        rbi=90,
    )


def _pitching_stats(
    player_id: str = "p1",
    name: str = "Pitcher",
    ip: float = 180.0,
    er: int = 60,
    so: int = 200,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name=name,
        year=2024,
        age=27,
        ip=ip,
        g=32,
        gs=32,
        er=er,
        h=150,
        bb=50,
        so=so,
        hr=20,
        hbp=5,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _fake_batting_source(data: dict[int, list[BattingSeasonStats]]) -> Any:
    """Create a fake DataSource[BattingSeasonStats] callable."""

    def source(query: Any) -> Ok[list[BattingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        return Ok(data.get(get_context().year, []))

    return source


def _fake_pitching_source(data: dict[int, list[PitchingSeasonStats]]) -> Any:
    """Create a fake DataSource[PitchingSeasonStats] callable."""

    def source(query: Any) -> Ok[list[PitchingSeasonStats]]:
        from fantasy_baseball_manager.context import get_context

        return Ok(data.get(get_context().year, []))

    return source


@pytest.fixture(autouse=True)
def _setup_context() -> Generator[None]:
    init_context(year=2024)
    yield
    reset_context()


_DEFAULT_CONFIG = EvaluationConfig(
    year=2024,
    batting_categories=(StatCategory.HR, StatCategory.SB, StatCategory.OBP),
    pitching_categories=(StatCategory.K, StatCategory.ERA, StatCategory.WHIP),
    min_pa=0,
    min_ip=0.0,
    top_n=20,
)


class TestEvaluateSourceInnerJoin:
    def test_only_matched_players_included(self) -> None:
        """Players in projections but not actuals (and vice versa) are excluded."""
        projected_batters = [
            _batter_proj(player_id="b1", hr=30.0),
            _batter_proj(player_id="b2", hr=20.0),  # not in actuals
        ]
        actual_batters = [
            _batting_stats(player_id="b1", hr=28),
            _batting_stats(player_id="b3", hr=15),  # not in projections
        ]
        source = SimpleProjectionSource(_batting=projected_batters, _pitching=[])
        batting_src = _fake_batting_source({2024: actual_batters})
        pitching_src = _fake_pitching_source({2024: []})
        result = evaluate_source(source, "test", batting_src, pitching_src, _DEFAULT_CONFIG)
        # HR stat accuracy should have sample_size=1 (only b1 matched)
        hr_acc = next(sa for sa in result.batting_stat_accuracy if sa.category == StatCategory.HR)
        assert hr_acc.sample_size == 1

    def test_min_pa_threshold_applied(self) -> None:
        """Actuals below min_pa should be filtered out before join."""
        projected = [_batter_proj(player_id="b1", hr=30.0)]
        actuals = [_batting_stats(player_id="b1", pa=100, hr=28)]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=200,
            min_ip=0.0,
            top_n=20,
        )
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        result = evaluate_source(source, "test", batting_src, pitching_src, config)
        assert result.batting_stat_accuracy == ()


class TestEvaluateSourceStatAccuracy:
    def test_perfect_prediction(self) -> None:
        """When projected == actual, RMSE and MAE should be 0, correlation should be 0 (constant)."""
        projected = [
            _batter_proj(player_id="b1", hr=25.0),
            _batter_proj(player_id="b2", hr=30.0),
            _batter_proj(player_id="b3", hr=35.0),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=25),
            _batting_stats(player_id="b2", hr=30),
            _batting_stats(player_id="b3", hr=35),
        ]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
        )
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        result = evaluate_source(source, "test", batting_src, pitching_src, config)
        hr_acc = result.batting_stat_accuracy[0]
        assert hr_acc.rmse == pytest.approx(0.0, abs=1e-10)
        assert hr_acc.mae == pytest.approx(0.0, abs=1e-10)
        assert hr_acc.correlation == pytest.approx(1.0)


class TestEvaluateSourceRankAccuracy:
    def test_rank_accuracy_computed(self) -> None:
        """With enough matched players, rank accuracy should be computed."""
        projected = [
            _batter_proj(player_id="b1", hr=40.0, sb=5.0),
            _batter_proj(player_id="b2", hr=30.0, sb=15.0),
            _batter_proj(player_id="b3", hr=20.0, sb=25.0),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=38, sb=6),
            _batting_stats(player_id="b2", hr=28, sb=14),
            _batting_stats(player_id="b3", hr=22, sb=24),
        ]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        result = evaluate_source(source, "test", batting_src, pitching_src, _DEFAULT_CONFIG)
        assert result.batting_rank_accuracy is not None
        assert result.batting_rank_accuracy.sample_size == 3

    def test_fewer_than_two_players_returns_none(self) -> None:
        """With < 2 matched players, rank accuracy should be None."""
        projected = [_batter_proj(player_id="b1", hr=30.0)]
        actuals = [_batting_stats(player_id="b1", hr=28)]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        result = evaluate_source(source, "test", batting_src, pitching_src, _DEFAULT_CONFIG)
        assert result.batting_rank_accuracy is None


class TestEvaluateSourceNoOverlap:
    def test_no_overlap_returns_empty_accuracy(self) -> None:
        projected = [_batter_proj(player_id="b1")]
        actuals = [_batting_stats(player_id="b99")]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        result = evaluate_source(source, "test", batting_src, pitching_src, _DEFAULT_CONFIG)
        assert result.batting_stat_accuracy == ()
        assert result.batting_rank_accuracy is None


class TestEvaluate:
    def test_multiple_sources(self) -> None:
        projected_a = [
            _batter_proj(player_id="b1", hr=30.0),
            _batter_proj(player_id="b2", hr=25.0),
        ]
        projected_b = [
            _batter_proj(player_id="b1", hr=28.0),
            _batter_proj(player_id="b2", hr=27.0),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=29),
            _batting_stats(player_id="b2", hr=26),
        ]
        source_a = SimpleProjectionSource(_batting=projected_a, _pitching=[])
        source_b = SimpleProjectionSource(_batting=projected_b, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
        )
        result = evaluate(
            sources=[("source_a", source_a), ("source_b", source_b)],
            batting_source=batting_src,
            pitching_source=pitching_src,
            config=config,
        )
        assert len(result.evaluations) == 2
        assert result.evaluations[0].source_name == "source_a"
        assert result.evaluations[1].source_name == "source_b"

    def test_end_to_end_pitching(self) -> None:
        projected = [
            _pitcher_proj(player_id="p1", so=200.0, ip=180.0, er=60.0),
            _pitcher_proj(player_id="p2", so=150.0, ip=160.0, er=70.0),
            _pitcher_proj(player_id="p3", so=250.0, ip=200.0, er=50.0),
        ]
        actuals = [
            _pitching_stats(player_id="p1", so=190, ip=175.0, er=62),
            _pitching_stats(player_id="p2", so=155, ip=158.0, er=68),
            _pitching_stats(player_id="p3", so=240, ip=195.0, er=52),
        ]
        source = SimpleProjectionSource(_batting=[], _pitching=projected)
        batting_src = _fake_batting_source({2024: []})
        pitching_src = _fake_pitching_source({2024: actuals})
        result = evaluate_source(source, "test", batting_src, pitching_src, _DEFAULT_CONFIG)
        assert len(result.pitching_stat_accuracy) == 3
        assert result.pitching_rank_accuracy is not None


class TestStratification:
    def test_pa_bucket_stratification(self) -> None:
        """Players are correctly grouped by PA bucket."""
        # Create players with different PA: 250 (bucket 200-400), 450 (bucket 400-600), 700 (bucket 600-1500)
        projected = [
            _batter_proj(player_id="b1", hr=30.0, pa=250.0),
            _batter_proj(player_id="b2", hr=25.0, pa=250.0),
            _batter_proj(player_id="b3", hr=20.0, pa=450.0),
            _batter_proj(player_id="b4", hr=35.0, pa=450.0),
            _batter_proj(player_id="b5", hr=40.0, pa=700.0),
            _batter_proj(player_id="b6", hr=28.0, pa=700.0),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=28, pa=250),
            _batting_stats(player_id="b2", hr=24, pa=250),
            _batting_stats(player_id="b3", hr=22, pa=450),
            _batting_stats(player_id="b4", hr=33, pa=450),
            _batting_stats(player_id="b5", hr=38, pa=700),
            _batting_stats(player_id="b6", hr=30, pa=700),
        ]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
            stratification=StratificationConfig(),
        )
        result = evaluate_source(source, "test", batting_src, pitching_src, config)
        assert len(result.batting_strata) > 0
        # Check that PA buckets exist
        pa_strata = [s for s in result.batting_strata if s.stratum_name.startswith("PA")]
        assert len(pa_strata) == 3
        # Check sample sizes
        for s in pa_strata:
            assert s.sample_size == 2

    def test_age_bucket_stratification(self) -> None:
        """Players are correctly grouped by age bucket."""
        projected = [
            _batter_proj(player_id="b1", hr=30.0, age=23),
            _batter_proj(player_id="b2", hr=25.0, age=24),
            _batter_proj(player_id="b3", hr=20.0, age=29),
            _batter_proj(player_id="b4", hr=35.0, age=30),
            _batter_proj(player_id="b5", hr=40.0, age=35),
            _batter_proj(player_id="b6", hr=28.0, age=38),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=28, age=23),
            _batting_stats(player_id="b2", hr=24, age=24),
            _batting_stats(player_id="b3", hr=22, age=29),
            _batting_stats(player_id="b4", hr=33, age=30),
            _batting_stats(player_id="b5", hr=38, age=35),
            _batting_stats(player_id="b6", hr=30, age=38),
        ]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
            stratification=StratificationConfig(),
        )
        result = evaluate_source(source, "test", batting_src, pitching_src, config)
        age_strata = [s for s in result.batting_strata if s.stratum_name.startswith("AGE")]
        assert len(age_strata) == 3
        for s in age_strata:
            assert s.sample_size == 2

    def test_empty_bucket_excluded(self) -> None:
        """Buckets with < 2 players are not included in strata."""
        # Only players in PA 600-1500 bucket
        projected = [
            _batter_proj(player_id="b1", hr=30.0, pa=700.0),
            _batter_proj(player_id="b2", hr=25.0, pa=650.0),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=28, pa=700),
            _batting_stats(player_id="b2", hr=24, pa=650),
        ]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
            stratification=StratificationConfig(),
        )
        result = evaluate_source(source, "test", batting_src, pitching_src, config)
        pa_strata = [s for s in result.batting_strata if s.stratum_name.startswith("PA")]
        # Only the 600-1500 bucket should be included
        assert len(pa_strata) == 1
        assert pa_strata[0].stratum_name == "PA 600-1500"

    def test_residuals_computed_when_requested(self) -> None:
        """PlayerResidual objects are populated when include_residuals=True."""
        projected = [
            _batter_proj(player_id="b1", hr=30.0),
            _batter_proj(player_id="b2", hr=25.0),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=28),
            _batting_stats(player_id="b2", hr=27),
        ]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
            stratification=StratificationConfig(include_residuals=True),
        )
        result = evaluate_source(source, "test", batting_src, pitching_src, config)
        assert result.batting_residuals is not None
        assert len(result.batting_residuals) == 2  # 2 players * 1 category
        # Check residual values
        b1_residual = next(r for r in result.batting_residuals if r.player_id == "b1")
        assert b1_residual.projected == 30.0
        assert b1_residual.actual == 28.0
        assert b1_residual.residual == -2.0
        assert b1_residual.abs_residual == 2.0

    def test_no_stratification_when_config_none(self) -> None:
        """Existing behavior preserved when stratification=None."""
        projected = [
            _batter_proj(player_id="b1", hr=30.0),
            _batter_proj(player_id="b2", hr=25.0),
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=28),
            _batting_stats(player_id="b2", hr=27),
        ]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
            stratification=None,
        )
        result = evaluate_source(source, "test", batting_src, pitching_src, config)
        assert result.batting_strata == ()
        assert result.batting_residuals is None


class TestHeadToHead:
    def test_compare_sources_wins_counted(self) -> None:
        """Head-to-head correctly counts wins for each source."""
        # Source A is better for b1 (error 2), Source B is better for b2 (error 1)
        projected_a = [
            _batter_proj(player_id="b1", hr=30.0),  # error: 2
            _batter_proj(player_id="b2", hr=20.0),  # error: 7
        ]
        projected_b = [
            _batter_proj(player_id="b1", hr=35.0),  # error: 7
            _batter_proj(player_id="b2", hr=26.0),  # error: 1
        ]
        actuals = [
            _batting_stats(player_id="b1", hr=28),
            _batting_stats(player_id="b2", hr=27),
        ]
        source_a = SimpleProjectionSource(_batting=projected_a, _pitching=[])
        source_b = SimpleProjectionSource(_batting=projected_b, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
            stratification=StratificationConfig(include_residuals=True),
        )
        result_a = evaluate_source(source_a, "source_a", batting_src, pitching_src, config)
        result_b = evaluate_source(source_b, "source_b", batting_src, pitching_src, config)

        h2h_results = compare_sources(result_a, result_b)
        assert len(h2h_results) == 1
        h2h = h2h_results[0]
        assert h2h.source_a == "source_a"
        assert h2h.source_b == "source_b"
        assert h2h.category == StatCategory.HR
        assert h2h.sample_size == 2
        assert h2h.a_wins == 1  # Source A wins for b1
        assert h2h.b_wins == 1  # Source B wins for b2
        assert h2h.ties == 0

    def test_compare_requires_residuals(self) -> None:
        """compare_sources raises ValueError if residuals not included."""
        projected = [_batter_proj(player_id="b1", hr=30.0)]
        actuals = [_batting_stats(player_id="b1", hr=28)]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        batting_src = _fake_batting_source({2024: actuals})
        pitching_src = _fake_pitching_source({2024: []})
        config = EvaluationConfig(
            year=2024,
            batting_categories=(StatCategory.HR,),
            pitching_categories=(),
            min_pa=0,
            min_ip=0.0,
            top_n=20,
            stratification=None,  # No residuals
        )
        result_a = evaluate_source(source, "a", batting_src, pitching_src, config)
        result_b = evaluate_source(source, "b", batting_src, pitching_src, config)

        with pytest.raises(ValueError, match="Residuals required"):
            compare_sources(result_a, result_b)
