import pytest

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.models import SGPDenominators, StatCategory
from fantasy_baseball_manager.valuation.sgp import compute_sgp_denominators, sgp_batting, sgp_pitching


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


class TestComputeSGPDenominators:
    def test_basic_denominator(self) -> None:
        # 12 teams, standings from 100 to 210 (best) for HR
        standings = {
            StatCategory.HR: [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0]
        }
        result = compute_sgp_denominators(standings, team_count=12)
        # denom = (210 - 100) / (12 - 1) = 110 / 11 = 10.0
        assert result.denominators[StatCategory.HR] == pytest.approx(10.0)

    def test_multiple_categories(self) -> None:
        standings = {
            StatCategory.HR: [100.0, 150.0, 200.0],
            StatCategory.SB: [50.0, 75.0, 100.0],
        }
        result = compute_sgp_denominators(standings, team_count=3)
        assert result.denominators[StatCategory.HR] == pytest.approx(50.0)
        assert result.denominators[StatCategory.SB] == pytest.approx(25.0)

    def test_single_team_raises(self) -> None:
        standings = {StatCategory.HR: [100.0]}
        with pytest.raises(ValueError, match="at least 2"):
            compute_sgp_denominators(standings, team_count=1)


class TestSGPBatting:
    def test_counting_stat(self) -> None:
        batters = [
            _batter(player_id="a", hr=10.0),
            _batter(player_id="b", hr=20.0),
            _batter(player_id="c", hr=30.0),
        ]
        denoms = SGPDenominators(denominators={StatCategory.HR: 10.0})
        results = sgp_batting(batters, (StatCategory.HR,), denoms)
        by_id = {pv.player_id: pv for pv in results}
        # Mean HR = 20. SGP = (stat - mean) / denom
        # a: (10-20)/10 = -1.0, b: 0.0, c: +1.0
        assert by_id["a"].total_value == pytest.approx(-1.0)
        assert by_id["b"].total_value == pytest.approx(0.0)
        assert by_id["c"].total_value == pytest.approx(1.0)

    def test_obp_volume_weighted(self) -> None:
        batters = [
            _batter(player_id="a", h=180.0, bb=60.0, hbp=10.0, pa=600.0),
            _batter(player_id="b", h=120.0, bb=40.0, hbp=5.0, pa=600.0),
        ]
        denoms = SGPDenominators(denominators={StatCategory.OBP: 5.0})
        results = sgp_batting(batters, (StatCategory.OBP,), denoms)
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value > 0
        assert by_id["b"].total_value < 0

    def test_empty_list(self) -> None:
        denoms = SGPDenominators(denominators={StatCategory.HR: 10.0})
        assert sgp_batting([], (StatCategory.HR,), denoms) == []

    def test_category_values_populated(self) -> None:
        batters = [_batter(player_id="a", hr=25.0)]
        denoms = SGPDenominators(denominators={StatCategory.HR: 10.0})
        results = sgp_batting(batters, (StatCategory.HR,), denoms)
        assert len(results[0].category_values) == 1
        assert results[0].category_values[0].category == StatCategory.HR
        assert results[0].category_values[0].raw_stat == 25.0


class TestSGPPitching:
    def test_counting_stat(self) -> None:
        pitchers = [
            _pitcher(player_id="a", so=150.0),
            _pitcher(player_id="b", so=200.0),
            _pitcher(player_id="c", so=250.0),
        ]
        denoms = SGPDenominators(denominators={StatCategory.K: 50.0})
        results = sgp_pitching(pitchers, (StatCategory.K,), denoms)
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value == pytest.approx(-1.0)
        assert by_id["b"].total_value == pytest.approx(0.0)
        assert by_id["c"].total_value == pytest.approx(1.0)

    def test_era_lower_is_better(self) -> None:
        pitchers = [
            _pitcher(player_id="a", er=40.0, ip=200.0),
            _pitcher(player_id="b", er=80.0, ip=200.0),
        ]
        denoms = SGPDenominators(denominators={StatCategory.ERA: 5.0})
        results = sgp_pitching(pitchers, (StatCategory.ERA,), denoms)
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value > 0
        assert by_id["b"].total_value < 0

    def test_whip_lower_is_better(self) -> None:
        pitchers = [
            _pitcher(player_id="a", h=150.0, bb=30.0, ip=200.0),
            _pitcher(player_id="b", h=200.0, bb=70.0, ip=200.0),
        ]
        denoms = SGPDenominators(denominators={StatCategory.WHIP: 5.0})
        results = sgp_pitching(pitchers, (StatCategory.WHIP,), denoms)
        by_id = {pv.player_id: pv for pv in results}
        assert by_id["a"].total_value > 0
        assert by_id["b"].total_value < 0

    def test_empty_list(self) -> None:
        denoms = SGPDenominators(denominators={StatCategory.K: 50.0})
        assert sgp_pitching([], (StatCategory.K,), denoms) == []
