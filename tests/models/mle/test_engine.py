import pytest

from fantasy_baseball_manager.domain.league_environment import LeagueEnvironment
from fantasy_baseball_manager.domain.level_factor import LevelFactor
from fantasy_baseball_manager.domain.minor_league_batting_stats import (
    MinorLeagueBattingStats,
)
from fantasy_baseball_manager.models.mle.engine import (
    apply_recency_weights,
    combine_translated_lines,
    compute_competition_factor,
    regress_to_mlb,
    translate_batting_line,
    translate_rates,
)
from fantasy_baseball_manager.models.mle.types import (
    AgeAdjustmentConfig,
    DEFAULT_AGE_BENCHMARKS,
    MLEConfig,
    TranslatedBattingLine,
)


class TestComputeCompetitionFactor:
    def test_known_values(self) -> None:
        m = compute_competition_factor(milb_rpg=5.0, mlb_rpg=4.5, level_factor=0.80)
        expected = (5.0 / 4.5) * 0.80
        assert m == pytest.approx(expected)

    def test_equal_environments(self) -> None:
        m = compute_competition_factor(milb_rpg=4.5, mlb_rpg=4.5, level_factor=0.75)
        assert m == pytest.approx(0.75)

    def test_lower_scoring_milb(self) -> None:
        m = compute_competition_factor(milb_rpg=4.0, mlb_rpg=4.5, level_factor=0.80)
        assert m < 0.80


class TestTranslateRates:
    """Tests for translate_rates helper."""

    def _default_config(self) -> MLEConfig:
        return MLEConfig()

    def test_k_pct_increases(self) -> None:
        # k_factor=1.12 * k_experience_factor=1.15 > 1 → K% goes up
        result = translate_rates(
            k_pct=0.200,
            bb_pct=0.080,
            iso=0.180,
            babip=0.320,
            pa=500,
            k_factor=1.12,
            bb_factor=0.88,
            iso_factor=0.72,
            mlb_babip=0.300,
            config=self._default_config(),
        )
        assert result.k_pct > 0.200

    def test_bb_pct_decreases(self) -> None:
        result = translate_rates(
            k_pct=0.200,
            bb_pct=0.080,
            iso=0.180,
            babip=0.320,
            pa=500,
            k_factor=1.12,
            bb_factor=0.88,
            iso_factor=0.72,
            mlb_babip=0.300,
            config=self._default_config(),
        )
        assert result.bb_pct < 0.080

    def test_iso_decreases(self) -> None:
        result = translate_rates(
            k_pct=0.200,
            bb_pct=0.080,
            iso=0.180,
            babip=0.320,
            pa=500,
            k_factor=1.12,
            bb_factor=0.88,
            iso_factor=0.72,
            mlb_babip=0.300,
            config=self._default_config(),
        )
        assert result.iso < 0.180

    def test_babip_regresses_toward_mlb_average(self) -> None:
        # .320 BABIP should regress toward .300
        result = translate_rates(
            k_pct=0.200,
            bb_pct=0.080,
            iso=0.180,
            babip=0.320,
            pa=500,
            k_factor=1.12,
            bb_factor=0.88,
            iso_factor=0.72,
            mlb_babip=0.300,
            config=self._default_config(),
        )
        assert 0.300 < result.babip < 0.320

    def test_babip_low_pa_regresses_heavily(self) -> None:
        # 120 PA → approx_bip is small → heavy regression toward league avg
        result = translate_rates(
            k_pct=0.200,
            bb_pct=0.080,
            iso=0.180,
            babip=0.350,
            pa=120,
            k_factor=1.12,
            bb_factor=0.88,
            iso_factor=0.72,
            mlb_babip=0.300,
            config=self._default_config(),
        )
        # Should be very close to .300
        assert abs(result.babip - 0.300) < 0.010

    def test_babip_high_pa_retains_more(self) -> None:
        config = self._default_config()
        low_pa = translate_rates(
            k_pct=0.200,
            bb_pct=0.080,
            iso=0.180,
            babip=0.350,
            pa=150,
            k_factor=1.12,
            bb_factor=0.88,
            iso_factor=0.72,
            mlb_babip=0.300,
            config=config,
        )
        high_pa = translate_rates(
            k_pct=0.200,
            bb_pct=0.080,
            iso=0.180,
            babip=0.350,
            pa=600,
            k_factor=1.12,
            bb_factor=0.88,
            iso_factor=0.72,
            mlb_babip=0.300,
            config=config,
        )
        # Higher PA retains more player signal → further from MLB avg
        assert high_pa.babip > low_pa.babip


def _aa_stats(
    *,
    pa: int = 500,
    ab: int = 450,
    h: int = 135,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 20,
    bb: int = 40,
    so: int = 100,
    hbp: int | None = 5,
    sf: int | None = 5,
    age: float = 22.0,
) -> MinorLeagueBattingStats:
    return MinorLeagueBattingStats(
        player_id=1,
        season=2025,
        level="AA",
        league="Eastern League",
        team="Binghamton Rumble Ponies",
        g=120,
        pa=pa,
        ab=ab,
        h=h,
        doubles=doubles,
        triples=triples,
        hr=hr,
        r=70,
        rbi=65,
        bb=bb,
        so=so,
        sb=10,
        cs=3,
        avg=h / ab if ab else 0.0,
        obp=0.340,
        slg=0.520,
        age=age,
        hbp=hbp,
        sf=sf,
    )


def _aa_level_factor() -> LevelFactor:
    return LevelFactor(
        level="AA",
        season=2025,
        factor=0.68,
        k_factor=1.12,
        bb_factor=0.88,
        iso_factor=0.72,
        babip_factor=0.90,
    )


def _aaa_level_factor() -> LevelFactor:
    return LevelFactor(
        level="AAA",
        season=2025,
        factor=0.80,
        k_factor=1.05,
        bb_factor=0.94,
        iso_factor=0.85,
        babip_factor=0.95,
    )


def _milb_env() -> LeagueEnvironment:
    return LeagueEnvironment(
        league="Eastern League",
        season=2025,
        level="AA",
        runs_per_game=5.2,
        avg=0.260,
        obp=0.330,
        slg=0.410,
        k_pct=0.220,
        bb_pct=0.085,
        hr_per_pa=0.030,
        babip=0.310,
    )


def _mlb_env() -> LeagueEnvironment:
    return LeagueEnvironment(
        league="MLB",
        season=2025,
        level="MLB",
        runs_per_game=4.6,
        avg=0.248,
        obp=0.315,
        slg=0.400,
        k_pct=0.230,
        bb_pct=0.083,
        hr_per_pa=0.035,
        babip=0.300,
    )


class TestTranslateBattingLine:
    def test_hand_calculated_full_line(self) -> None:
        result = translate_batting_line(
            stats=_aa_stats(),
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        assert result.pa == 500
        assert result.ab == 455
        assert result.h == 114
        assert result.doubles == 26
        assert result.triples == 3
        assert result.hr == 15
        assert result.bb == 35
        assert result.so == 129
        assert result.hbp == 5
        assert result.sf == 5
        assert result.avg == pytest.approx(114 / 455, abs=1e-4)
        assert result.obp == pytest.approx(154 / 500, abs=1e-4)
        assert result.slg == pytest.approx(191 / 455, abs=1e-4)
        assert result.k_pct == pytest.approx(129 / 500, abs=1e-4)
        assert result.bb_pct == pytest.approx(35 / 500, abs=1e-4)

    def test_k_pct_increases_after_translation(self) -> None:
        stats = _aa_stats()
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        assert result.k_pct > stats.so / stats.pa

    def test_iso_decreases_after_translation(self) -> None:
        stats = _aa_stats()
        original_iso = (235 / 450) - (135 / 450)  # TB/AB - H/AB
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        assert result.iso < original_iso

    def test_babip_regresses_toward_mlb(self) -> None:
        stats = _aa_stats()
        original_babip = (135 - 20) / (450 - 100 - 20 + 5)
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        assert 0.300 < result.babip < original_babip

    def test_zero_hr_player(self) -> None:
        stats = _aa_stats(hr=0, h=115, ab=450)
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        assert result.hr == 0
        assert result.pa == 500

    def test_very_high_babip(self) -> None:
        # .400 BABIP player with H inflated
        stats = _aa_stats(h=165, ab=450)
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        # Should be heavily regressed toward .300
        assert result.babip < 0.370

    def test_very_low_babip(self) -> None:
        stats = _aa_stats(h=100, ab=450)
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        # Low BABIP gets pulled up toward MLB avg
        original_babip = (100 - 20) / (450 - 100 - 20 + 5)
        assert result.babip > original_babip

    def test_below_min_pa_raises_value_error(self) -> None:
        stats = _aa_stats(pa=50, ab=45, h=12, bb=4, so=10)
        with pytest.raises(ValueError, match="minimum"):
            translate_batting_line(
                stats=stats,
                league_env=_milb_env(),
                mlb_env=_mlb_env(),
                level_factor=_aa_level_factor(),
                config=MLEConfig(),
            )

    def test_none_hbp_sf_treated_as_zero(self) -> None:
        stats = _aa_stats(hbp=None, sf=None)
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        assert result.hbp == 0
        assert result.sf == 0
        assert result.pa == 500


class TestLevelOrderingAndInvariants:
    def _translate(self, level_factor: LevelFactor, milb_env: LeagueEnvironment | None = None) -> TranslatedBattingLine:
        return translate_batting_line(
            stats=_aa_stats(),
            league_env=milb_env or _milb_env(),
            mlb_env=_mlb_env(),
            level_factor=level_factor,
            config=MLEConfig(),
        )

    def test_aaa_produces_better_line_than_aa(self) -> None:
        aaa_env = LeagueEnvironment(
            league="International League",
            season=2025,
            level="AAA",
            runs_per_game=5.0,
            avg=0.260,
            obp=0.335,
            slg=0.420,
            k_pct=0.215,
            bb_pct=0.088,
            hr_per_pa=0.032,
            babip=0.305,
        )
        aa_result = self._translate(_aa_level_factor())
        aaa_result = self._translate(_aaa_level_factor(), milb_env=aaa_env)
        assert aaa_result.avg > aa_result.avg
        assert aaa_result.obp > aa_result.obp
        assert aaa_result.slg > aa_result.slg
        assert aaa_result.k_pct < aa_result.k_pct

    def test_pa_equals_ab_plus_bb_plus_hbp_plus_sf(self) -> None:
        result = self._translate(_aa_level_factor())
        assert result.pa == result.ab + result.bb + result.hbp + result.sf

    def test_hits_geq_extra_base_hits(self) -> None:
        result = self._translate(_aa_level_factor())
        assert result.h >= result.doubles + result.triples + result.hr

    def test_avg_equals_h_over_ab(self) -> None:
        result = self._translate(_aa_level_factor())
        assert result.avg == pytest.approx(result.h / result.ab, abs=1e-6)

    def test_obp_formula(self) -> None:
        result = self._translate(_aa_level_factor())
        expected = (result.h + result.bb + result.hbp) / (result.ab + result.bb + result.hbp + result.sf)
        assert result.obp == pytest.approx(expected, abs=1e-6)

    def test_iso_equals_slg_minus_avg(self) -> None:
        result = self._translate(_aa_level_factor())
        assert result.iso == pytest.approx(result.slg - result.avg, abs=1e-6)

    def test_no_negative_counting_stats(self) -> None:
        result = self._translate(_aa_level_factor())
        assert result.pa >= 0
        assert result.ab >= 0
        assert result.h >= 0
        assert result.doubles >= 0
        assert result.triples >= 0
        assert result.hr >= 0
        assert result.bb >= 0
        assert result.so >= 0
        assert result.hbp >= 0
        assert result.sf >= 0


def _age_config() -> AgeAdjustmentConfig:
    return AgeAdjustmentConfig(benchmarks=DEFAULT_AGE_BENCHMARKS)


class TestAgeAdjustedTranslation:
    def test_age_adjusted_young_player_better_than_old(self) -> None:
        young_stats = _aa_stats(age=20.0)
        old_stats = _aa_stats(age=25.0)
        age_config = _age_config()

        young_result = translate_batting_line(
            stats=young_stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
            age_config=age_config,
        )
        old_result = translate_batting_line(
            stats=old_stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
            age_config=age_config,
        )
        assert young_result.avg > old_result.avg
        assert young_result.obp > old_result.obp
        assert young_result.slg > old_result.slg

    def test_age_adjustment_none_matches_unadjusted(self) -> None:
        stats = _aa_stats()
        unadjusted = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        explicit_none = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
            age_config=None,
        )
        assert unadjusted == explicit_none

    def test_age_adjustment_dampened_on_k_pct(self) -> None:
        # K% proportional change should be less than BABIP proportional change
        young_stats = _aa_stats(age=20.0)
        age_config = _age_config()

        adjusted = translate_batting_line(
            stats=young_stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
            age_config=age_config,
        )
        unadjusted = translate_batting_line(
            stats=young_stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        # BABIP gets full adjustment, K% gets dampened (half-strength)
        # Compare relative changes: BABIP relative change > K% relative change
        babip_relative = abs(adjusted.babip - unadjusted.babip) / unadjusted.babip
        k_relative = abs(adjusted.k_pct - unadjusted.k_pct) / unadjusted.k_pct
        assert babip_relative > k_relative

    def test_k_dampening_configurable(self) -> None:
        stats = _aa_stats(age=20.0)
        default_config = _age_config()  # k_dampening defaults to 0.5
        custom_config = AgeAdjustmentConfig(
            benchmarks=DEFAULT_AGE_BENCHMARKS,
            k_dampening=0.3,
        )

        default_result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
            age_config=default_config,
        )
        custom_result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
            age_config=custom_config,
        )

        # Different k_dampening should produce different K%
        assert default_result.k_pct != custom_result.k_pct

        # k_dampening=0.3 means less adjustment effect on K%
        # For young player (full_adj > 1), smaller dampening → K% closer to unadjusted
        unadjusted = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
        )
        assert abs(custom_result.k_pct - unadjusted.k_pct) < abs(default_result.k_pct - unadjusted.k_pct)

    def test_age_adjusted_line_preserves_invariants(self) -> None:
        stats = _aa_stats(age=20.0)
        age_config = _age_config()
        result = translate_batting_line(
            stats=stats,
            league_env=_milb_env(),
            mlb_env=_mlb_env(),
            level_factor=_aa_level_factor(),
            config=MLEConfig(),
            age_config=age_config,
        )
        # PA identity
        assert result.pa == result.ab + result.bb + result.hbp + result.sf
        # H >= XBH
        assert result.h >= result.doubles + result.triples + result.hr
        # No negative stats
        assert result.pa >= 0
        assert result.ab >= 0
        assert result.h >= 0
        assert result.doubles >= 0
        assert result.triples >= 0
        assert result.hr >= 0
        assert result.bb >= 0
        assert result.so >= 0


def _translated_line(
    *,
    player_id: int = 1,
    season: int = 2025,
    source_level: str = "AA",
    pa: int = 300,
    ab: int = 270,
    h: int = 70,
    doubles: int = 15,
    triples: int = 2,
    hr: int = 10,
    bb: int = 25,
    so: int = 75,
    hbp: int = 3,
    sf: int = 2,
) -> TranslatedBattingLine:
    return TranslatedBattingLine(
        player_id=player_id,
        season=season,
        source_level=source_level,
        pa=pa,
        ab=ab,
        h=h,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
    )


class TestCombineTranslatedLines:
    def test_combine_single_line(self) -> None:
        line = _translated_line()
        result = combine_translated_lines([line])
        assert result == line

    def test_combine_multi_level_pa_weights_rates(self) -> None:
        line_aa = _translated_line(pa=200, ab=180, h=50, so=60, bb=15, hbp=3, sf=2, doubles=10, triples=1, hr=5)
        line_aaa = _translated_line(
            pa=300, ab=270, h=70, so=60, bb=25, hbp=3, sf=2, doubles=15, triples=2, hr=8, source_level="AAA"
        )
        result = combine_translated_lines([line_aa, line_aaa])

        # Rates derived from summed counting stats
        # k_pct = total_so / total_pa = 120/500
        assert result.k_pct == pytest.approx(120 / 500, abs=1e-6)
        # bb_pct = total_bb / total_pa = 40/500
        assert result.bb_pct == pytest.approx(40 / 500, abs=1e-6)
        # iso = slg - avg, derived from summed counts
        total_tb = 120 + 25 + 3 * 2 + 13 * 3
        assert result.iso == pytest.approx(total_tb / 450 - 120 / 450, abs=1e-6)
        # babip = (H-HR) / (AB-SO-HR+SF) = (120-13)/(450-120-13+4) = 107/321
        assert result.babip == pytest.approx(107 / 321, abs=1e-6)

    def test_combine_sums_counting_stats(self) -> None:
        line1 = _translated_line(pa=200, ab=180, h=50, doubles=10, triples=1, hr=5, bb=15, so=50, hbp=3, sf=2)
        line2 = _translated_line(pa=300, ab=270, h=70, doubles=15, triples=2, hr=8, bb=25, so=60, hbp=3, sf=2)
        result = combine_translated_lines([line1, line2])

        assert result.pa == 500
        assert result.ab == 450
        assert result.h == 120
        assert result.doubles == 25
        assert result.triples == 3
        assert result.hr == 13
        assert result.bb == 40
        assert result.so == 110
        assert result.hbp == 6
        assert result.sf == 4

    def test_combine_derives_rates_from_totals(self) -> None:
        line1 = _translated_line(pa=200, ab=180, h=50, doubles=10, triples=1, hr=5, bb=15, so=50, hbp=3, sf=2)
        line2 = _translated_line(pa=300, ab=270, h=70, doubles=15, triples=2, hr=8, bb=25, so=60, hbp=3, sf=2)
        result = combine_translated_lines([line1, line2])

        # avg = 120/450
        assert result.avg == pytest.approx(120 / 450, abs=1e-6)
        # obp = (120+40+6) / (450+40+6+4)
        assert result.obp == pytest.approx(166 / 500, abs=1e-6)
        # slg = (120 + 25 + 3*2 + 13*3) / 450
        tb = 120 + 25 + 3 * 2 + 13 * 3
        assert result.slg == pytest.approx(tb / 450, abs=1e-6)


class TestApplyRecencyWeights:
    def test_recency_single_season(self) -> None:
        line = _translated_line()
        result = apply_recency_weights([(line, 5.0)])
        # Default: so=75, pa=300, bb=25
        assert result.k_pct == pytest.approx(75 / 300, abs=1e-6)
        assert result.bb_pct == pytest.approx(25 / 300, abs=1e-6)

    def test_recency_recent_season_weighted_higher(self) -> None:
        recent = _translated_line(pa=400, ab=360, h=70, so=80, bb=30, hbp=5, sf=5)
        older = _translated_line(pa=400, ab=360, h=70, so=120, bb=30, hbp=5, sf=5)
        result = apply_recency_weights([(recent, 5.0), (older, 3.0)])

        # With equal PA, rates should be closer to recent line
        # weighted_so = round(80*5 + 120*3) = 760
        # weighted_pa = round(400*5 + 400*3) = 3200
        # k_pct = 760/3200 = 0.2375
        assert result.k_pct == pytest.approx(760 / 3200, abs=1e-6)
        assert result.k_pct < 0.250  # Closer to 0.200 than 0.300

    def test_recency_zero_weight_season_ignored(self) -> None:
        active = _translated_line(pa=400, ab=360, h=80, so=88, bb=30, hbp=5, sf=5)
        ignored = _translated_line(pa=400, ab=360, h=60, so=140, bb=30, hbp=5, sf=5)
        result = apply_recency_weights([(active, 5.0), (ignored, 0.0)])

        # Only one active line → returned unchanged
        assert result.k_pct == pytest.approx(88 / 400, abs=1e-6)
        assert result.bb_pct == pytest.approx(30 / 400, abs=1e-6)


class TestRegressToMlb:
    def test_regress_high_pa_retains_rates(self) -> None:
        rates = {"k_pct": 0.200, "bb_pct": 0.100, "iso": 0.200, "babip": 0.320}
        mlb_rates = {"k_pct": 0.230, "bb_pct": 0.083, "iso": 0.160, "babip": 0.300}
        result = regress_to_mlb(rates, mlb_rates, effective_pa=600.0, regression_pa=1200.0)

        # (0.200*600 + 0.230*1200) / 1800 ≈ 0.220
        assert result["k_pct"] == pytest.approx((0.200 * 600 + 0.230 * 1200) / 1800, abs=1e-6)
        # Player rate has 1/3 weight, so result is between player and league
        assert 0.200 < result["k_pct"] < 0.230

    def test_regress_low_pa_moves_toward_league(self) -> None:
        rates = {"k_pct": 0.200, "bb_pct": 0.100}
        mlb_rates = {"k_pct": 0.230, "bb_pct": 0.083}
        result = regress_to_mlb(rates, mlb_rates, effective_pa=100.0, regression_pa=1200.0)

        # With 100 effective PA vs 1200 regression, result is mostly league avg
        assert result["k_pct"] > 0.225  # Very close to 0.230

    def test_regress_zero_pa_equals_league(self) -> None:
        rates = {"k_pct": 0.200, "bb_pct": 0.100, "iso": 0.200}
        mlb_rates = {"k_pct": 0.230, "bb_pct": 0.083, "iso": 0.160}
        result = regress_to_mlb(rates, mlb_rates, effective_pa=0.0, regression_pa=1200.0)

        assert result["k_pct"] == pytest.approx(0.230, abs=1e-6)
        assert result["bb_pct"] == pytest.approx(0.083, abs=1e-6)
        assert result["iso"] == pytest.approx(0.160, abs=1e-6)
