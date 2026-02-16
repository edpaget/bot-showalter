import pytest

from fantasy_baseball_manager.models.mle.blend import blend_mle_with_mlb, blend_rate
from fantasy_baseball_manager.models.mle.types import BlendConfig


class TestBlendRate:
    def test_known_values(self) -> None:
        """Hand-calculated 3-way blend: MLB + MLE + regression."""
        # 200 MLB PA at .250, 400 MLE PA at .200, discount 0.55,
        # regression 120 PA, league .220
        # effective_mle = 400 * 0.55 = 220
        # num = 200*.250 + 220*.200 + 120*.220 = 50 + 44 + 26.4 = 120.4
        # den = 200 + 220 + 120 = 540
        # result = 120.4 / 540 ≈ 0.22296
        result = blend_rate(
            mlb_pa=200,
            mlb_rate=0.250,
            mle_pa=400,
            mle_rate=0.200,
            discount=0.55,
            regression_pa=120.0,
            league_rate=0.220,
        )
        assert result == pytest.approx(120.4 / 540, abs=1e-6)

    def test_zero_mlb_pa(self) -> None:
        """Pure MLE player — MLE + regression only."""
        # 400 MLE PA at .200, discount 0.55 → effective 220
        # num = 0 + 220*.200 + 120*.220 = 44 + 26.4 = 70.4
        # den = 0 + 220 + 120 = 340
        result = blend_rate(
            mlb_pa=0,
            mlb_rate=0.0,
            mle_pa=400,
            mle_rate=0.200,
            discount=0.55,
            regression_pa=120.0,
            league_rate=0.220,
        )
        assert result == pytest.approx(70.4 / 340, abs=1e-6)

    def test_zero_mle_pa(self) -> None:
        """MLB only — MLB + regression."""
        # 500 MLB PA at .260, no MLE
        # num = 500*.260 + 120*.220 = 130 + 26.4 = 156.4
        # den = 500 + 0 + 120 = 620
        result = blend_rate(
            mlb_pa=500,
            mlb_rate=0.260,
            mle_pa=0,
            mle_rate=0.0,
            discount=0.55,
            regression_pa=120.0,
            league_rate=0.220,
        )
        assert result == pytest.approx(156.4 / 620, abs=1e-6)

    def test_zero_both(self) -> None:
        """No data at all — returns league average."""
        result = blend_rate(
            mlb_pa=0,
            mlb_rate=0.0,
            mle_pa=0,
            mle_rate=0.0,
            discount=0.55,
            regression_pa=120.0,
            league_rate=0.220,
        )
        assert result == pytest.approx(0.220, abs=1e-6)


LEAGUE_RATES: dict[str, float] = {
    "k_pct": 0.220,
    "bb_pct": 0.080,
    "iso": 0.150,
    "babip": 0.300,
}


def _make_mle_stats(
    player_id: int = 1,
    season: int = 2025,
    pa: int = 400,
    k_pct: float = 0.180,
    bb_pct: float = 0.090,
    iso: float = 0.160,
    babip: float = 0.310,
) -> dict[str, object]:
    """Build a stat dict matching MLE projection stat_json format."""
    ab = round(pa * (1 - bb_pct) - 5)  # rough estimate for ab
    so = round(pa * k_pct)
    bb = round(pa * bb_pct)
    hr = round(ab * iso / 3.0)
    bip = ab - so - hr
    h_from_bip = round(babip * bip) if bip > 0 else 0
    h = h_from_bip + hr
    doubles = round(h * 0.20)
    triples = round(h * 0.03)
    return {
        "pa": pa,
        "ab": ab,
        "h": h,
        "doubles": doubles,
        "triples": triples,
        "hr": hr,
        "bb": bb,
        "so": so,
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "iso": iso,
        "babip": babip,
    }


def _make_mlb_season(
    pa: int = 200,
    k_pct: float = 0.200,
    bb_pct: float = 0.085,
    iso: float = 0.155,
    babip: float = 0.305,
) -> dict[str, float]:
    """Build a dict of MLB season rates + PA."""
    return {
        "pa": float(pa),
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "iso": iso,
        "babip": babip,
    }


class TestBlendMleWithMlb:
    def test_pure_mle_heavily_regressed(self) -> None:
        """0 MLB PA → BABIP regressed heavily toward league avg."""
        mle_stats = _make_mle_stats(pa=400, babip=0.350)
        config = BlendConfig()

        result = blend_mle_with_mlb(
            mlb_seasons=[],
            mle_stats=mle_stats,
            config=config,
            league_rates=LEAGUE_RATES,
        )

        assert result.mlb_pa == 0
        assert result.mle_pa == 400
        assert result.effective_pa == pytest.approx(400 * 0.55)
        # BABIP should be pulled heavily toward .300 (820 PA regression)
        assert result.rates["babip"] < 0.340
        assert result.rates["babip"] > LEAGUE_RATES["babip"]

    def test_mixed_weights_both(self) -> None:
        """200 MLB PA + 400 MLE PA, effective_pa correct."""
        mlb_seasons = [_make_mlb_season(pa=200, k_pct=0.200)]
        mle_stats = _make_mle_stats(pa=400, k_pct=0.180)
        config = BlendConfig()

        result = blend_mle_with_mlb(
            mlb_seasons=mlb_seasons,
            mle_stats=mle_stats,
            config=config,
            league_rates=LEAGUE_RATES,
        )

        assert result.mlb_pa == 200
        assert result.mle_pa == 400
        assert result.effective_pa == pytest.approx(200 + 400 * 0.55)
        # K% blended should be between MLB and MLE values
        assert 0.170 < result.rates["k_pct"] < 0.210

    def test_established_player_mle_negligible(self) -> None:
        """2000 MLB PA — MLE barely changes result."""
        mlb_seasons = [
            _make_mlb_season(pa=600, k_pct=0.200, bb_pct=0.085, iso=0.155, babip=0.305),
            _make_mlb_season(pa=600, k_pct=0.200, bb_pct=0.085, iso=0.155, babip=0.305),
            _make_mlb_season(pa=500, k_pct=0.200, bb_pct=0.085, iso=0.155, babip=0.305),
            _make_mlb_season(pa=300, k_pct=0.200, bb_pct=0.085, iso=0.155, babip=0.305),
        ]
        mle_stats = _make_mle_stats(pa=400, k_pct=0.150)
        config = BlendConfig()

        result = blend_mle_with_mlb(
            mlb_seasons=mlb_seasons,
            mle_stats=mle_stats,
            config=config,
            league_rates=LEAGUE_RATES,
        )

        assert result.mlb_pa == 2000
        # K% should be very close to MLB .200, barely pulled by MLE .150
        assert result.rates["k_pct"] == pytest.approx(0.200, abs=0.010)

    def test_component_specific_stabilization(self) -> None:
        """K% retains more signal than BABIP at same sample size."""
        mle_stats = _make_mle_stats(pa=200, k_pct=0.150, babip=0.350)
        config = BlendConfig()

        result = blend_mle_with_mlb(
            mlb_seasons=[],
            mle_stats=mle_stats,
            config=config,
            league_rates=LEAGUE_RATES,
        )

        # K% stabilizes at 60 PA → 110/(110+60) = ~65% signal retained
        # BABIP stabilizes at 820 PA → 110/(110+820) = ~12% signal retained
        effective = 200 * 0.55  # = 110

        k_signal_weight = effective / (effective + 60.0)
        babip_signal_weight = effective / (effective + 820.0)

        assert k_signal_weight > babip_signal_weight
        # K% closer to player's true rate than BABIP is
        k_distance_from_league = abs(result.rates["k_pct"] - LEAGUE_RATES["k_pct"])
        babip_distance_from_league = abs(result.rates["babip"] - LEAGUE_RATES["babip"])
        # Relative to the gap, K% preserves more
        k_gap = abs(0.150 - LEAGUE_RATES["k_pct"])  # .070
        babip_gap = abs(0.350 - LEAGUE_RATES["babip"])  # .050
        k_retained = k_distance_from_league / k_gap if k_gap > 0 else 0
        babip_retained = babip_distance_from_league / babip_gap if babip_gap > 0 else 0
        assert k_retained > babip_retained
