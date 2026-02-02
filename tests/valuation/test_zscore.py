import pytest

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.models import StatCategory
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching


def _batter(
    player_id: str = "b1",
    name: str = "Batter",
    pa: float = 600.0,
    ab: float = 540.0,
    h: float = 150.0,
    hr: float = 30.0,
    bb: float = 50.0,
    hbp: float = 5.0,
    sb: float = 10.0,
) -> BattingProjection:
    return BattingProjection(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        pa=pa,
        ab=ab,
        h=h,
        singles=h - 40.0,
        doubles=25.0,
        triples=5.0,
        hr=hr,
        bb=bb,
        so=100.0,
        hbp=hbp,
        sf=3.0,
        sh=2.0,
        sb=sb,
        cs=3.0,
        r=0.0,
        rbi=0.0,
    )


def _pitcher(
    player_id: str = "p1",
    name: str = "Pitcher",
    ip: float = 200.0,
    er: float = 60.0,
    h: float = 170.0,
    bb: float = 50.0,
    so: float = 180.0,
) -> PitchingProjection:
    return PitchingProjection(
        player_id=player_id,
        name=name,
        year=2025,
        age=27,
        ip=ip,
        g=33.0,
        gs=33.0,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=20.0,
        hbp=8.0,
        era=er / ip * 9 if ip != 0.0 else 0.0,
        whip=(h + bb) / ip if ip != 0.0 else 0.0,
    )


class TestZscoreBattingCounting:
    def test_single_counting_category(self) -> None:
        batters = [
            _batter(player_id="a", hr=10.0),
            _batter(player_id="b", hr=20.0),
            _batter(player_id="c", hr=30.0),
        ]
        results = zscore_batting(batters, (StatCategory.HR,))
        by_id = {pv.player_id: pv for pv in results}
        # Mean=20, std=~8.165
        assert by_id["a"].total_value < 0
        assert by_id["b"].total_value == pytest.approx(0.0, abs=1e-10)
        assert by_id["c"].total_value > 0

    def test_multiple_counting_categories(self) -> None:
        batters = [
            _batter(player_id="a", hr=10.0, sb=20.0),
            _batter(player_id="b", hr=20.0, sb=10.0),
        ]
        results = zscore_batting(batters, (StatCategory.HR, StatCategory.SB))
        by_id = {pv.player_id: pv for pv in results}
        # HR: a=-1, b=+1; SB: a=+1, b=-1 -> both sum to 0
        assert by_id["a"].total_value == pytest.approx(0.0, abs=1e-10)
        assert by_id["b"].total_value == pytest.approx(0.0, abs=1e-10)

    def test_category_values_present(self) -> None:
        batters = [_batter(player_id="a", hr=10.0), _batter(player_id="b", hr=30.0)]
        results = zscore_batting(batters, (StatCategory.HR,))
        for pv in results:
            assert len(pv.category_values) == 1
            assert pv.category_values[0].category == StatCategory.HR

    def test_single_player_returns_zero(self) -> None:
        results = zscore_batting([_batter(hr=30.0)], (StatCategory.HR,))
        assert results[0].total_value == 0.0

    def test_empty_list(self) -> None:
        results = zscore_batting([], (StatCategory.HR,))
        assert results == []

    def test_identical_stats_zero_std(self) -> None:
        batters = [_batter(player_id="a", hr=20.0), _batter(player_id="b", hr=20.0)]
        results = zscore_batting(batters, (StatCategory.HR,))
        for pv in results:
            assert pv.total_value == 0.0


class TestZscoreBattingRatio:
    def test_obp_volume_weighted(self) -> None:
        # Player A: high OBP, high PA -> should get positive z
        # Player B: low OBP, high PA -> should get negative z
        batters = [
            _batter(player_id="a", h=180.0, bb=60.0, hbp=10.0, pa=600.0),
            _batter(player_id="b", h=120.0, bb=40.0, hbp=5.0, pa=600.0),
        ]
        results = zscore_batting(batters, (StatCategory.OBP,))
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value > 0
        assert by_id["b"].total_value < 0

    def test_obp_pa_matters(self) -> None:
        # Same OBP rate but different PA -> different contribution magnitude
        # Player A: 200/600 = .333, Player B: 100/300 = .333
        # Pool avg OBP = (200+100)/(600+300) = 300/900 = .333
        # Contributions: A = 200 - 600*.333 = 0.2, B = 100 - 300*.333 = 0.1
        # Both near zero since rate matches pool average
        batters = [
            _batter(player_id="a", h=150.0, bb=45.0, hbp=5.0, pa=600.0),
            _batter(player_id="b", h=75.0, bb=22.5, hbp=2.5, pa=300.0),
        ]
        results = zscore_batting(batters, (StatCategory.OBP,))
        by_id = {pv.player_id: pv for pv in results}
        # Same rate as pool avg -> contributions near zero -> z-scores near zero
        assert by_id["a"].total_value == pytest.approx(0.0, abs=1e-6)
        assert by_id["b"].total_value == pytest.approx(0.0, abs=1e-6)


class TestZscorePitchingCounting:
    def test_strikeouts(self) -> None:
        pitchers = [
            _pitcher(player_id="a", so=150.0),
            _pitcher(player_id="b", so=200.0),
            _pitcher(player_id="c", so=250.0),
        ]
        results = zscore_pitching(pitchers, (StatCategory.K,))
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value < 0
        assert by_id["b"].total_value == pytest.approx(0.0, abs=1e-10)
        assert by_id["c"].total_value > 0

    def test_empty_list(self) -> None:
        assert zscore_pitching([], (StatCategory.K,)) == []


class TestZscorePitchingRatio:
    def test_era_lower_is_better(self) -> None:
        pitchers = [
            _pitcher(player_id="a", er=40.0, ip=200.0),  # low ERA
            _pitcher(player_id="b", er=80.0, ip=200.0),  # high ERA
        ]
        results = zscore_pitching(pitchers, (StatCategory.ERA,))
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value > 0  # low ERA = positive value
        assert by_id["b"].total_value < 0  # high ERA = negative value

    def test_whip_lower_is_better(self) -> None:
        pitchers = [
            _pitcher(player_id="a", h=150.0, bb=30.0, ip=200.0),  # low WHIP
            _pitcher(player_id="b", h=200.0, bb=70.0, ip=200.0),  # high WHIP
        ]
        results = zscore_pitching(pitchers, (StatCategory.WHIP,))
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value > 0
        assert by_id["b"].total_value < 0
