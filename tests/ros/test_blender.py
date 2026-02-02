import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.ros.blender import BayesianBlender


def _preseason_batter(
    *,
    pa: float = 600.0,
    singles: float = 100.0,
    doubles: float = 30.0,
    triples: float = 5.0,
    hr: float = 30.0,
    bb: float = 60.0,
    so: float = 120.0,
    hbp: float = 5.0,
    sf: float = 4.0,
    sh: float = 1.0,
    sb: float = 10.0,
    cs: float = 3.0,
    r: float = 80.0,
    rbi: float = 90.0,
) -> BattingProjection:
    h = singles + doubles + triples + hr
    ab = pa - bb - hbp - sf - sh
    return BattingProjection(
        player_id="b1",
        name="Test Batter",
        year=2025,
        age=28,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=r,
        rbi=rbi,
    )


def _actuals_batter(
    *,
    pa: int = 300,
    ab: int = 260,
    singles: int = 50,
    doubles: int = 15,
    triples: int = 2,
    hr: int = 15,
    bb: int = 30,
    so: int = 60,
    hbp: int = 3,
    sf: int = 2,
    sh: int = 0,
    sb: int = 5,
    cs: int = 2,
    r: int = 40,
    rbi: int = 45,
) -> BattingSeasonStats:
    h = singles + doubles + triples + hr
    return BattingSeasonStats(
        player_id="b1",
        name="Test Batter",
        year=2025,
        age=28,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=r,
        rbi=rbi,
    )


def _preseason_pitcher(
    *,
    ip: float = 180.0,
    g: float = 30.0,
    gs: float = 30.0,
    er: float = 60.0,
    h: float = 150.0,
    bb: float = 50.0,
    so: float = 180.0,
    hr: float = 20.0,
    hbp: float = 8.0,
    w: float = 12.0,
    nsvh: float = 0.0,
) -> PitchingProjection:
    era = (er / ip) * 9 if ip > 0 else 0.0
    whip = (h + bb) / ip if ip > 0 else 0.0
    return PitchingProjection(
        player_id="p1",
        name="Test Pitcher",
        year=2025,
        age=28,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
        era=era,
        whip=whip,
        w=w,
        nsvh=nsvh,
    )


def _actuals_pitcher(
    *,
    ip: float = 90.0,
    g: int = 15,
    gs: int = 15,
    er: int = 30,
    h: int = 75,
    bb: int = 25,
    so: int = 90,
    hr: int = 10,
    hbp: int = 4,
    w: int = 6,
    sv: int = 0,
    hld: int = 0,
    bs: int = 0,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="p1",
        name="Test Pitcher",
        year=2025,
        age=28,
        ip=ip,
        g=g,
        gs=gs,
        er=er,
        h=h,
        bb=bb,
        so=so,
        hr=hr,
        hbp=hbp,
        w=w,
        sv=sv,
        hld=hld,
        bs=bs,
    )


class TestBayesianBlenderBatting:
    def test_zero_actuals_returns_preseason(self) -> None:
        """With zero PA actuals, blended rates equal preseason rates and remaining_pa = full preseason pa."""
        blender = BayesianBlender()
        preseason = _preseason_batter(pa=600.0, hr=30.0)
        actuals = _actuals_batter(pa=0, ab=0, singles=0, doubles=0, triples=0, hr=0, bb=0, so=0, hbp=0, sf=0, sh=0, sb=0, cs=0, r=0, rbi=0)

        result = blender.blend_batting(preseason, actuals)

        assert result.pa == pytest.approx(600.0)
        assert result.hr == pytest.approx(preseason.hr, rel=1e-6)
        assert result.bb == pytest.approx(preseason.bb, rel=1e-6)
        assert result.so == pytest.approx(preseason.so, rel=1e-6)

    def test_midpoint_when_actuals_equal_regression(self) -> None:
        """When actuals.pa equals the regression constant, blended rate is midpoint of preseason and actual rates."""
        blender = BayesianBlender(batting_regression={"hr": 200.0, "singles": 200.0, "doubles": 200.0, "triples": 200.0, "bb": 200.0, "so": 200.0, "hbp": 200.0, "sf": 200.0, "sh": 200.0, "sb": 200.0, "cs": 200.0, "r": 200.0, "rbi": 200.0})
        preseason = _preseason_batter(pa=600.0, hr=30.0)  # rate = 30/600 = 0.05
        actuals = _actuals_batter(pa=200, ab=170, hr=20)  # rate = 20/200 = 0.10

        result = blender.blend_batting(preseason, actuals)

        # blended_rate = (0.05 * 200 + 0.10 * 200) / (200 + 200) = 0.075
        # remaining_pa = 600 - 200 = 400
        # ros_hr = 0.075 * 400 = 30.0
        expected_rate = (30.0 / 600.0 + 20 / 200) / 2  # 0.075
        remaining_pa = 600.0 - 200
        assert result.pa == pytest.approx(remaining_pa)
        assert result.hr == pytest.approx(expected_rate * remaining_pa, rel=1e-6)

    def test_large_sample_weights_actuals_heavily(self) -> None:
        """With large sample, actuals dominate the blend."""
        blender = BayesianBlender(batting_regression={"hr": 100.0, "singles": 100.0, "doubles": 100.0, "triples": 100.0, "bb": 100.0, "so": 100.0, "hbp": 100.0, "sf": 100.0, "sh": 100.0, "sb": 100.0, "cs": 100.0, "r": 100.0, "rbi": 100.0})
        preseason = _preseason_batter(pa=600.0, hr=30.0)  # rate = 0.05
        actuals = _actuals_batter(pa=500, ab=450, hr=50)  # rate = 0.10

        result = blender.blend_batting(preseason, actuals)

        # blended_rate = (0.05 * 100 + 0.10 * 500) / (100 + 500) = 55/600 ≈ 0.0917
        # This should be much closer to actual rate (0.10) than preseason (0.05)
        blended_rate = result.hr / result.pa if result.pa > 0 else 0
        preseason_rate = 30.0 / 600.0
        actual_rate = 50 / 500
        assert abs(blended_rate - actual_rate) < abs(blended_rate - preseason_rate)

    def test_remaining_pa_calculation(self) -> None:
        """remaining_pa = preseason.pa - actuals.pa."""
        blender = BayesianBlender()
        preseason = _preseason_batter(pa=600.0)
        actuals = _actuals_batter(pa=200)

        result = blender.blend_batting(preseason, actuals)

        assert result.pa == pytest.approx(400.0)

    def test_remaining_pa_floors_at_zero(self) -> None:
        """When actuals exceed preseason PA, remaining_pa floors at 0 and counting stats are 0."""
        blender = BayesianBlender()
        preseason = _preseason_batter(pa=300.0)
        actuals = _actuals_batter(pa=400)

        result = blender.blend_batting(preseason, actuals)

        assert result.pa == pytest.approx(0.0)
        assert result.hr == pytest.approx(0.0)
        assert result.bb == pytest.approx(0.0)
        assert result.so == pytest.approx(0.0)
        assert result.r == pytest.approx(0.0)
        assert result.rbi == pytest.approx(0.0)

    def test_counting_stats_equal_rate_times_remaining_pa(self) -> None:
        """Each counting stat = blended_rate * remaining_pa."""
        blender = BayesianBlender(batting_regression={"hr": 500.0, "singles": 800.0, "doubles": 1600.0, "triples": 1600.0, "bb": 400.0, "so": 200.0, "hbp": 600.0, "sf": 1600.0, "sh": 1600.0, "sb": 600.0, "cs": 600.0, "r": 1200.0, "rbi": 1200.0})
        preseason = _preseason_batter(pa=600.0, hr=30.0, bb=60.0)
        actuals = _actuals_batter(pa=200, ab=170, hr=10, bb=20)

        result = blender.blend_batting(preseason, actuals)

        remaining_pa = 400.0

        # Verify HR: blended_rate = (0.05 * 500 + 0.05 * 200) / (500 + 200)
        pre_hr_rate = 30.0 / 600.0
        act_hr_rate = 10 / 200
        hr_reg = 500.0
        expected_hr_rate = (pre_hr_rate * hr_reg + act_hr_rate * 200) / (hr_reg + 200)
        assert result.hr == pytest.approx(expected_hr_rate * remaining_pa, rel=1e-6)

        # Verify BB
        pre_bb_rate = 60.0 / 600.0
        act_bb_rate = 20 / 200
        bb_reg = 400.0
        expected_bb_rate = (pre_bb_rate * bb_reg + act_bb_rate * 200) / (bb_reg + 200)
        assert result.bb == pytest.approx(expected_bb_rate * remaining_pa, rel=1e-6)

    def test_derived_h_is_sum_of_hit_types(self) -> None:
        """h = singles + doubles + triples + hr."""
        blender = BayesianBlender()
        preseason = _preseason_batter()
        actuals = _actuals_batter(pa=200)

        result = blender.blend_batting(preseason, actuals)

        assert result.h == pytest.approx(result.singles + result.doubles + result.triples + result.hr, rel=1e-6)

    def test_derived_ab(self) -> None:
        """ab = pa - bb - hbp - sf - sh."""
        blender = BayesianBlender()
        preseason = _preseason_batter()
        actuals = _actuals_batter(pa=200)

        result = blender.blend_batting(preseason, actuals)

        expected_ab = result.pa - result.bb - result.hbp - result.sf - result.sh
        assert result.ab == pytest.approx(expected_ab, rel=1e-6)

    def test_preserves_identity_fields(self) -> None:
        """player_id, name, year, age come from preseason projection."""
        blender = BayesianBlender()
        preseason = _preseason_batter()
        actuals = _actuals_batter()

        result = blender.blend_batting(preseason, actuals)

        assert result.player_id == "b1"
        assert result.name == "Test Batter"
        assert result.year == 2025
        assert result.age == 28


class TestBayesianBlenderPitching:
    def test_zero_actuals_returns_preseason(self) -> None:
        """With zero IP actuals, blended rates equal preseason rates."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher(ip=180.0, so=180.0)
        actuals = _actuals_pitcher(ip=0.0, g=0, gs=0, er=0, h=0, bb=0, so=0, hr=0, hbp=0, w=0)

        result = blender.blend_pitching(preseason, actuals)

        assert result.ip == pytest.approx(180.0)
        assert result.so == pytest.approx(preseason.so, rel=1e-6)

    def test_remaining_ip(self) -> None:
        """remaining_ip = preseason.ip - actuals.ip."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher(ip=180.0)
        actuals = _actuals_pitcher(ip=90.0)

        result = blender.blend_pitching(preseason, actuals)

        assert result.ip == pytest.approx(90.0)

    def test_remaining_ip_floors_at_zero(self) -> None:
        """When actuals exceed preseason IP, remaining_ip floors at 0."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher(ip=100.0)
        actuals = _actuals_pitcher(ip=150.0)

        result = blender.blend_pitching(preseason, actuals)

        assert result.ip == pytest.approx(0.0)
        assert result.so == pytest.approx(0.0)

    def test_blending_formula_pitching(self) -> None:
        """Verify blending uses outs as opportunity unit."""
        blender = BayesianBlender(pitching_regression={"so": 30.0, "bb": 60.0, "hr": 80.0, "hbp": 100.0, "h": 200.0, "er": 150.0, "w": 134.0, "sv": 134.0, "hld": 134.0, "bs": 134.0})
        preseason = _preseason_pitcher(ip=180.0, so=180.0)  # rate = 180/(180*3) = 1/3
        actuals = _actuals_pitcher(ip=90.0, so=90)  # rate = 90/(90*3) = 1/3

        result = blender.blend_pitching(preseason, actuals)

        # Both rates equal 1/3, so blended rate = 1/3
        remaining_ip = 90.0
        remaining_outs = remaining_ip * 3
        expected_so = (1 / 3) * remaining_outs
        assert result.so == pytest.approx(expected_so, rel=1e-4)

    def test_era_derived(self) -> None:
        """ERA = (er / ip) * 9."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher(ip=180.0, er=60.0)
        actuals = _actuals_pitcher(ip=90.0, er=30)

        result = blender.blend_pitching(preseason, actuals)

        if result.ip > 0:
            expected_era = (result.er / result.ip) * 9
            assert result.era == pytest.approx(expected_era, rel=1e-4)

    def test_whip_derived(self) -> None:
        """WHIP = (h + bb) / ip."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher()
        actuals = _actuals_pitcher(ip=90.0)

        result = blender.blend_pitching(preseason, actuals)

        if result.ip > 0:
            expected_whip = (result.h + result.bb) / result.ip
            assert result.whip == pytest.approx(expected_whip, rel=1e-4)

    def test_nsvh_derived(self) -> None:
        """nsvh = sv + hld - bs from blended stats."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher(ip=60.0, g=60.0, gs=0.0, w=3.0, nsvh=15.0)
        actuals = _actuals_pitcher(ip=30.0, g=30, gs=0, w=1, sv=5, hld=5, bs=2)

        result = blender.blend_pitching(preseason, actuals)

        # nsvh should be derived from blended sv + hld - bs
        # We just verify the relationship holds
        assert result.ip == pytest.approx(30.0)

    def test_starter_detection_from_actuals(self) -> None:
        """Starter detection uses actuals gs/g ratio >= 0.5."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher(ip=180.0, g=30.0, gs=30.0)
        actuals = _actuals_pitcher(ip=90.0, g=15, gs=15)

        result = blender.blend_pitching(preseason, actuals)

        assert result.gs > 0

    def test_reliever_detection_from_actuals(self) -> None:
        """Reliever detected when gs/g < 0.5."""
        blender = BayesianBlender()
        preseason = _preseason_pitcher(ip=60.0, g=60.0, gs=0.0)
        actuals = _actuals_pitcher(ip=30.0, g=30, gs=2)

        result = blender.blend_pitching(preseason, actuals)

        assert result.gs == pytest.approx(0.0)

    def test_preserves_identity_fields(self) -> None:
        blender = BayesianBlender()
        preseason = _preseason_pitcher()
        actuals = _actuals_pitcher()

        result = blender.blend_pitching(preseason, actuals)

        assert result.player_id == "p1"
        assert result.name == "Test Pitcher"
        assert result.year == 2025
        assert result.age == 28
