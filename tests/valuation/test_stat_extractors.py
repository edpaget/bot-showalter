import pytest

from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.valuation.models import StatCategory
from fantasy_baseball_manager.valuation.stat_extractors import extract_batting_stat, extract_pitching_stat


def _batting(
    *,
    pa: float = 600.0,
    ab: float = 540.0,
    h: float = 150.0,
    hr: float = 30.0,
    bb: float = 50.0,
    hbp: float = 5.0,
    sb: float = 10.0,
) -> BattingProjection:
    return BattingProjection(
        player_id="b1",
        name="Batter",
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


def _pitching(
    *,
    ip: float = 200.0,
    er: float = 60.0,
    h: float = 170.0,
    bb: float = 50.0,
    so: float = 180.0,
) -> PitchingProjection:
    return PitchingProjection(
        player_id="p1",
        name="Pitcher",
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


class TestExtractBattingStat:
    def test_hr(self) -> None:
        assert extract_batting_stat(_batting(hr=30.0), StatCategory.HR) == 30.0

    def test_sb(self) -> None:
        assert extract_batting_stat(_batting(sb=15.0), StatCategory.SB) == 15.0

    def test_obp(self) -> None:
        proj = _batting(h=150.0, bb=50.0, hbp=5.0, pa=600.0)
        expected = (150.0 + 50.0 + 5.0) / 600.0
        assert extract_batting_stat(proj, StatCategory.OBP) == pytest.approx(expected)

    def test_obp_zero_pa(self) -> None:
        proj = _batting(h=0.0, bb=0.0, hbp=0.0, pa=0.0)
        assert extract_batting_stat(proj, StatCategory.OBP) == 0.0

    def test_r(self) -> None:
        proj = _batting()
        proj = BattingProjection(
            player_id="b1",
            name="Batter",
            year=2025,
            age=28,
            pa=600.0,
            ab=540.0,
            h=150.0,
            singles=110.0,
            doubles=25.0,
            triples=5.0,
            hr=30.0,
            bb=50.0,
            so=100.0,
            hbp=5.0,
            sf=3.0,
            sh=2.0,
            sb=10.0,
            cs=3.0,
            r=85.0,
            rbi=95.0,
        )
        assert extract_batting_stat(proj, StatCategory.R) == 85.0

    def test_rbi(self) -> None:
        proj = BattingProjection(
            player_id="b1",
            name="Batter",
            year=2025,
            age=28,
            pa=600.0,
            ab=540.0,
            h=150.0,
            singles=110.0,
            doubles=25.0,
            triples=5.0,
            hr=30.0,
            bb=50.0,
            so=100.0,
            hbp=5.0,
            sf=3.0,
            sh=2.0,
            sb=10.0,
            cs=3.0,
            r=85.0,
            rbi=95.0,
        )
        assert extract_batting_stat(proj, StatCategory.RBI) == 95.0

    def test_pitching_category_raises(self) -> None:
        with pytest.raises(ValueError, match="ERA"):
            extract_batting_stat(_batting(), StatCategory.ERA)


class TestExtractPitchingStat:
    def test_k(self) -> None:
        assert extract_pitching_stat(_pitching(so=180.0), StatCategory.K) == 180.0

    def test_era(self) -> None:
        proj = _pitching(er=60.0, ip=200.0)
        expected = 60.0 / 200.0 * 9
        assert extract_pitching_stat(proj, StatCategory.ERA) == pytest.approx(expected)

    def test_era_zero_ip(self) -> None:
        proj = _pitching(er=0.0, ip=0.0)
        assert extract_pitching_stat(proj, StatCategory.ERA) == 0.0

    def test_whip(self) -> None:
        proj = _pitching(h=170.0, bb=50.0, ip=200.0)
        expected = (170.0 + 50.0) / 200.0
        assert extract_pitching_stat(proj, StatCategory.WHIP) == pytest.approx(expected)

    def test_whip_zero_ip(self) -> None:
        proj = _pitching(h=0.0, bb=0.0, ip=0.0)
        assert extract_pitching_stat(proj, StatCategory.WHIP) == 0.0

    def test_unsupported_w(self) -> None:
        with pytest.raises(ValueError, match="W"):
            extract_pitching_stat(_pitching(), StatCategory.W)

    def test_unsupported_nsvh(self) -> None:
        with pytest.raises(ValueError, match="NSVH"):
            extract_pitching_stat(_pitching(), StatCategory.NSVH)

    def test_batting_category_raises(self) -> None:
        with pytest.raises(ValueError, match="HR"):
            extract_pitching_stat(_pitching(), StatCategory.HR)
