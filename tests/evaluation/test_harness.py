import pytest

from fantasy_baseball_manager.evaluation.harness import (
    EvaluationConfig,
    evaluate,
    evaluate_source,
)
from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
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
) -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=name,
        year=2024,
        age=28,
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
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=2024,
        age=28,
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


class FakeDataSource:
    def __init__(
        self,
        batting: dict[int, list[BattingSeasonStats]],
        pitching: dict[int, list[PitchingSeasonStats]],
    ) -> None:
        self._batting = batting
        self._pitching = pitching

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return []

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return []


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
        ds = FakeDataSource(
            batting={2024: actual_batters},
            pitching={2024: []},
        )
        result = evaluate_source(source, "test", ds, _DEFAULT_CONFIG)
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
        ds = FakeDataSource(batting={2024: actuals}, pitching={2024: []})
        result = evaluate_source(source, "test", ds, config)
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
        ds = FakeDataSource(batting={2024: actuals}, pitching={2024: []})
        result = evaluate_source(source, "test", ds, config)
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
        ds = FakeDataSource(batting={2024: actuals}, pitching={2024: []})
        result = evaluate_source(source, "test", ds, _DEFAULT_CONFIG)
        assert result.batting_rank_accuracy is not None
        assert result.batting_rank_accuracy.sample_size == 3

    def test_fewer_than_two_players_returns_none(self) -> None:
        """With < 2 matched players, rank accuracy should be None."""
        projected = [_batter_proj(player_id="b1", hr=30.0)]
        actuals = [_batting_stats(player_id="b1", hr=28)]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        ds = FakeDataSource(batting={2024: actuals}, pitching={2024: []})
        result = evaluate_source(source, "test", ds, _DEFAULT_CONFIG)
        assert result.batting_rank_accuracy is None


class TestEvaluateSourceNoOverlap:
    def test_no_overlap_returns_empty_accuracy(self) -> None:
        projected = [_batter_proj(player_id="b1")]
        actuals = [_batting_stats(player_id="b99")]
        source = SimpleProjectionSource(_batting=projected, _pitching=[])
        ds = FakeDataSource(batting={2024: actuals}, pitching={2024: []})
        result = evaluate_source(source, "test", ds, _DEFAULT_CONFIG)
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
        ds = FakeDataSource(batting={2024: actuals}, pitching={2024: []})
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
            data_source=ds,
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
        ds = FakeDataSource(batting={2024: []}, pitching={2024: actuals})
        result = evaluate_source(source, "test", ds, _DEFAULT_CONFIG)
        assert len(result.pitching_stat_accuracy) == 3
        assert result.pitching_rank_accuracy is not None
