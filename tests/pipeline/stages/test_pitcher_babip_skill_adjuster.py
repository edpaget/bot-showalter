import pytest

from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
from fantasy_baseball_manager.pipeline.stages.pitcher_babip_skill_adjuster import (
    PitcherBabipSkillAdjuster,
    PitcherBabipSkillConfig,
)
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates
from tests.conftest import make_test_feature_store


class FakeBattedBallSource:
    def __init__(self, data: dict[int, list[PitcherBattedBallStats]]) -> None:
        self._data = data

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        return self._data.get(year, [])


def _make_pitcher(
    player_id: str = "fgp1",
    name: str = "Test Pitcher",
    rates: dict[str, float] | None = None,
    metadata: PlayerMetadata | None = None,
) -> PlayerRates:
    default_rates: dict[str, float] = {
        "h": 0.240,
        "hr": 0.030,
        "bb": 0.080,
        "so": 0.220,
        "hbp": 0.008,
        "er": 0.040,
    }
    if rates:
        default_rates.update(rates)
    default_metadata: PlayerMetadata = {
        "is_starter": True,
        "ip_per_year": 180.0,
        "expected_babip": 0.300,
        "expected_lob_pct": 0.73,
    }
    if metadata:
        default_metadata.update(metadata)
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        rates=default_rates,
        opportunities=200.0,
        metadata=default_metadata,
    )


def _make_batter(player_id: str = "fg1") -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name="Test Batter",
        year=2025,
        age=30,
        rates={"hr": 0.040, "singles": 0.140, "bb": 0.080, "so": 0.200},
        opportunities=600.0,
        metadata={"pa_per_year": [600.0]},
    )


GROUND_BALL_PITCHER = PitcherBattedBallStats(
    player_id="fgp1",
    name="Ground Ball Pitcher",
    year=2024,
    pa=700,
    gb_pct=0.60,
    fb_pct=0.20,
    ld_pct=0.20,
    iffb_pct=0.10,
)

FLY_BALL_PITCHER = PitcherBattedBallStats(
    player_id="fgp1",
    name="Fly Ball Pitcher",
    year=2024,
    pa=700,
    gb_pct=0.30,
    fb_pct=0.40,
    ld_pct=0.28,
    iffb_pct=0.10,
)

LEAGUE_AVG_PITCHER = PitcherBattedBallStats(
    player_id="fgp1",
    name="Average Pitcher",
    year=2024,
    pa=700,
    gb_pct=0.43,
    fb_pct=0.35,
    ld_pct=0.20,
    iffb_pct=0.10,
)

EXTREME_PITCHER = PitcherBattedBallStats(
    player_id="fgp1",
    name="Extreme Pitcher",
    year=2024,
    pa=700,
    gb_pct=0.70,
    fb_pct=0.10,
    ld_pct=0.10,
    iffb_pct=0.30,
)


class TestBabipComputation:
    def test_ground_ball_pitcher_lowers_babip(self) -> None:
        """GB%=0.60 vs league 0.43 -> x_babip < 0.300."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        x_babip = result[0].metadata["pitcher_x_babip"]
        assert isinstance(x_babip, float)
        # gb_coeff=-0.10 * (0.60-0.43) = -0.017, ld unchanged -> x_babip < 0.300
        assert x_babip < 0.300

    def test_fly_ball_high_ld_pitcher_raises_babip(self) -> None:
        """LD%=0.28 vs league 0.20 -> x_babip > 0.300."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [FLY_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        x_babip = result[0].metadata["pitcher_x_babip"]
        assert isinstance(x_babip, float)
        # ld_coeff=0.20 * (0.28-0.20) = +0.016, gb_coeff=-0.10*(0.30-0.43)=+0.013
        assert x_babip > 0.300

    def test_league_average_profile_near_base(self) -> None:
        """League-average profile -> x_babip ~ 0.300."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [LEAGUE_AVG_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        x_babip = result[0].metadata["pitcher_x_babip"]
        assert isinstance(x_babip, float)
        assert x_babip == pytest.approx(0.300, abs=1e-6)

    def test_extreme_values_clamped(self) -> None:
        """Extreme batted-ball profile gets clamped to [min_babip, max_babip]."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [EXTREME_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        x_babip = result[0].metadata["pitcher_x_babip"]
        assert isinstance(x_babip, float)
        assert config.min_babip <= x_babip <= config.max_babip


class TestHRateAdjustment:
    def test_weight_zero_h_rate_unchanged(self) -> None:
        """weight=0 -> H rate unchanged from normalization output."""
        config = PitcherBabipSkillConfig(blend_weight=0.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # With weight=0, new_babip = expected_babip, so h stays the same
        # h_new = (hr + expected_babip * (1 - hr - so)) / (1 - expected_babip)
        hr = pitcher.rates["hr"]
        so = pitcher.rates["so"]
        expected_babip = 0.300
        expected_h = (hr + expected_babip * (1.0 - hr - so)) / (1.0 - expected_babip)
        assert result[0].rates["h"] == pytest.approx(expected_h, abs=1e-6)

    def test_weight_one_reflects_skill_babip(self) -> None:
        """weight=1 -> H rate fully reflects skill BABIP."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        x_babip = result[0].metadata["pitcher_x_babip"]
        assert isinstance(x_babip, float)
        # h_new should use x_babip directly
        hr = pitcher.rates["hr"]
        so = pitcher.rates["so"]
        expected_h = (hr + x_babip * (1.0 - hr - so)) / (1.0 - x_babip)
        assert result[0].rates["h"] == pytest.approx(expected_h, abs=1e-6)

    def test_ground_ball_pitcher_lower_h_rate(self) -> None:
        """Ground-ball pitcher gets lower H rate than normalization alone would give."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # Compute what H rate would be with expected_babip=0.300 (normalization only)
        hr = pitcher.rates["hr"]
        so = pitcher.rates["so"]
        norm_babip = 0.300
        norm_h = (hr + norm_babip * (1.0 - hr - so)) / (1.0 - norm_babip)
        # Ground-ball pitcher -> lower BABIP -> lower H rate than normalization
        assert result[0].rates["h"] < norm_h


class TestERRecalculation:
    def test_er_recalculated_from_new_h_rate(self) -> None:
        """ER = (h_new - hr + bb + hbp) * (1 - lob) + hr."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])

        h_new = result[0].rates["h"]
        hr = pitcher.rates["hr"]
        bb = pitcher.rates["bb"]
        hbp = pitcher.rates["hbp"]
        expected_lob = 0.73

        baserunners = h_new - hr + bb + hbp
        expected_er = baserunners * (1.0 - expected_lob) + hr
        assert result[0].rates["er"] == pytest.approx(expected_er, abs=1e-6)

    def test_er_uses_lob_from_metadata(self) -> None:
        """ER recalculation uses expected_lob_pct from metadata."""
        config = PitcherBabipSkillConfig(blend_weight=1.0)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher(metadata={"expected_lob_pct": 0.80})
        result = adjuster.adjust([pitcher])

        h_new = result[0].rates["h"]
        hr = pitcher.rates["hr"]
        bb = pitcher.rates["bb"]
        hbp = pitcher.rates["hbp"]

        baserunners = h_new - hr + bb + hbp
        expected_er = baserunners * (1.0 - 0.80) + hr
        assert result[0].rates["er"] == pytest.approx(expected_er, abs=1e-6)


class TestPassthrough:
    def test_batter_passes_through_unchanged(self) -> None:
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_pitcher_below_min_pa_passes_through(self) -> None:
        low_pa = PitcherBattedBallStats(
            player_id="fgp1",
            name="Low PA Pitcher",
            year=2024,
            pa=100,
            gb_pct=0.60,
            fb_pct=0.20,
            ld_pct=0.20,
            iffb_pct=0.10,
        )
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [low_pa]}),
            ),
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates

    def test_pitcher_not_in_data_passes_through(self) -> None:
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: []}),
            ),
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates

    def test_missing_expected_babip_passes_through(self) -> None:
        """Pitcher without expected_babip metadata (normalization didn't run)."""
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
        )
        pitcher = _make_pitcher(metadata={"expected_babip": None})  # type: ignore[typeddict-item]
        # Remove expected_babip entirely
        meta: PlayerMetadata = dict(pitcher.metadata)  # type: ignore[assignment]
        del meta["expected_babip"]
        pitcher = PlayerRates(
            player_id=pitcher.player_id,
            name=pitcher.name,
            year=pitcher.year,
            age=pitcher.age,
            rates=pitcher.rates,
            opportunities=pitcher.opportunities,
            metadata=meta,
        )
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates

    def test_empty_list(self) -> None:
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({}),
            ),
        )
        result = adjuster.adjust([])
        assert result == []


class TestMetadata:
    def test_diagnostics_stored(self) -> None:
        config = PitcherBabipSkillConfig(blend_weight=0.40)
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert "pitcher_x_babip" in result[0].metadata
        assert result[0].metadata["pitcher_gb_pct"] == 0.60
        assert "pitcher_babip_skill_blended" in result[0].metadata

    def test_existing_metadata_preserved(self) -> None:
        adjuster = PitcherBabipSkillAdjuster(
            feature_store=make_test_feature_store(
                batted_ball_source=FakeBattedBallSource({2024: [GROUND_BALL_PITCHER]}),
            ),
        )
        pitcher = _make_pitcher(metadata={"custom_key": "custom_value"})  # type: ignore[typeddict-unknown-key]
        result = adjuster.adjust([pitcher])
        assert result[0].metadata["custom_key"] == "custom_value"  # type: ignore[typeddict-item]
        assert "pitcher_x_babip" in result[0].metadata
