import pytest

from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
from fantasy_baseball_manager.pipeline.stages.statcast_adjuster import (
    StatcastBlendConfig,
    StatcastRateAdjuster,
)
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates
from fantasy_baseball_manager.player.identity import Player


class FakeStatcastSource:
    def __init__(self, data: dict[int, list[StatcastBatterStats]]) -> None:
        self._data = data

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return self._data.get(year, [])


def _make_batter(
    player_id: str = "fg1",
    name: str = "Test Batter",
    rates: dict[str, float] | None = None,
    metadata: PlayerMetadata | None = None,
    mlbam_id: str | None = "mlb1",
) -> PlayerRates:
    default_rates = {
        "hr": 0.040,
        "singles": 0.140,
        "doubles": 0.040,
        "triples": 0.005,
        "bb": 0.080,
        "so": 0.200,
        "hbp": 0.010,
        "sf": 0.005,
        "sh": 0.000,
        "sb": 0.020,
        "r": 0.100,
        "rbi": 0.110,
    }
    if rates:
        default_rates.update(rates)
    default_metadata: PlayerMetadata = {"pa_per_year": [600.0]}
    if metadata:
        default_metadata.update(metadata)
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=30,
        rates=default_rates,
        opportunities=600.0,
        metadata=default_metadata,
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id=mlbam_id, name=name),
    )


def _make_pitcher(player_id: str = "fgp1") -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name="Test Pitcher",
        year=2025,
        age=28,
        rates={"er": 0.040, "so": 0.250, "bb": 0.070},
        opportunities=200.0,
        metadata={"is_starter": True},
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id="mlb1", name="Test Pitcher"),
    )


SAMPLE_STATCAST = StatcastBatterStats(
    player_id="mlb1",
    name="Test Batter",
    year=2024,
    pa=500,
    barrel_rate=0.15,
    hard_hit_rate=0.42,
    xwoba=0.360,
    xba=0.270,
    xslg=0.480,
)


class TestBatterBlending:
    """Verify blend formula: final = w * statcast + (1-w) * marcel."""

    def test_weight_zero_returns_marcel(self) -> None:
        config = StatcastBlendConfig(blend_weight=0.0)
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates["hr"] == pytest.approx(batter.rates["hr"])
        assert result[0].rates["singles"] == pytest.approx(batter.rates["singles"])
        assert result[0].rates["doubles"] == pytest.approx(batter.rates["doubles"])
        assert result[0].rates["triples"] == pytest.approx(batter.rates["triples"])

    def test_weight_one_returns_statcast_derived(self) -> None:
        config = StatcastBlendConfig(blend_weight=1.0)
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        # With weight=1.0 the blended rates should differ from marcel
        assert result[0].rates["hr"] != batter.rates["hr"]

    def test_partial_blend_interpolates(self) -> None:
        config = StatcastBlendConfig(blend_weight=0.5)
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        # HR should be between marcel and statcast-derived values
        marcel_hr = batter.rates["hr"]
        blended_hr = result[0].rates["hr"]
        # At weight=0.5, blended should be different from marcel
        assert blended_hr != pytest.approx(marcel_hr)

    def test_non_blended_rates_unchanged(self) -> None:
        config = StatcastBlendConfig(blend_weight=0.5)
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates["bb"] == batter.rates["bb"]
        assert result[0].rates["so"] == batter.rates["so"]
        assert result[0].rates["sb"] == batter.rates["sb"]
        assert result[0].rates["r"] == batter.rates["r"]
        assert result[0].rates["rbi"] == batter.rates["rbi"]


class TestNonBatterPassthrough:
    """Pitchers should pass through unchanged."""

    def test_pitcher_unchanged(self) -> None:
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source)
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates


class TestNoStatcastData:
    """Players without Statcast data pass through."""

    def test_missing_statcast_record_passes_through(self) -> None:
        source = FakeStatcastSource({2024: []})
        adjuster = StatcastRateAdjuster(source)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_low_pa_passes_through(self) -> None:
        low_pa = StatcastBatterStats(
            player_id="mlb1",
            name="Low PA Batter",
            year=2024,
            pa=50,
            barrel_rate=0.15,
            hard_hit_rate=0.42,
            xwoba=0.360,
            xba=0.270,
            xslg=0.480,
        )
        source = FakeStatcastSource({2024: [low_pa]})
        adjuster = StatcastRateAdjuster(source)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_unmapped_id_passes_through(self) -> None:
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source)
        batter = _make_batter(mlbam_id=None)
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates


class TestHrDerivation:
    """Barrel rate -> HR rate conversion."""

    def test_hr_derived_from_barrel_rate(self) -> None:
        config = StatcastBlendConfig(
            blend_weight=1.0,
            league_hr_per_barrel=0.245,
        )
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        # bip_rate = 1 - (bb + so + hbp + sf + sh) = 1 - (0.08 + 0.20 + 0.01 + 0.005 + 0.0) = 0.705
        # statcast_hr = 0.15 * 0.245 * 0.705 = 0.025897...
        expected_hr = 0.15 * 0.245 * 0.705
        assert result[0].rates["hr"] == pytest.approx(expected_hr, abs=1e-6)


class TestHitDecomposition:
    """xBA/xSLG -> singles/doubles/triples decomposition."""

    def test_preserves_existing_2b_3b_ratio(self) -> None:
        config = StatcastBlendConfig(blend_weight=1.0)
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter(rates={"doubles": 0.040, "triples": 0.010})
        result = adjuster.adjust([batter])
        # doubles should be 4x triples (80/20 ratio)
        if result[0].rates["triples"] > 0:
            ratio = result[0].rates["doubles"] / result[0].rates["triples"]
            assert ratio == pytest.approx(4.0, abs=0.01)

    def test_negative_iso_clamped_to_zero(self) -> None:
        # xSLG barely above xBA so non_hr_iso would be negative
        low_power = StatcastBatterStats(
            player_id="mlb1",
            name="Contact Hitter",
            year=2024,
            pa=500,
            barrel_rate=0.02,
            hard_hit_rate=0.30,
            xwoba=0.300,
            xba=0.280,
            xslg=0.290,
        )
        config = StatcastBlendConfig(blend_weight=1.0)
        source = FakeStatcastSource({2024: [low_power]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates["doubles"] >= 0
        assert result[0].rates["triples"] >= 0

    def test_default_doubles_share_when_no_xbh_history(self) -> None:
        config = StatcastBlendConfig(blend_weight=1.0, default_doubles_share=0.85)
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter(rates={"doubles": 0.0, "triples": 0.0})
        result = adjuster.adjust([batter])
        # With 85/15 default split
        if result[0].rates["doubles"] > 0 and result[0].rates["triples"] > 0:
            ratio = result[0].rates["doubles"] / result[0].rates["triples"]
            expected_ratio = 0.85 / 0.15
            assert ratio == pytest.approx(expected_ratio, abs=0.1)


class TestMetadata:
    """Diagnostics stored in metadata."""

    def test_blended_flag_set(self) -> None:
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].metadata["statcast_blended"] is True

    def test_xwoba_stored(self) -> None:
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].metadata["statcast_xwoba"] == 0.360

    def test_blend_weight_stored(self) -> None:
        config = StatcastBlendConfig(blend_weight=0.40)
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].metadata["blend_weight_used"] == 0.40

    def test_no_metadata_when_not_blended(self) -> None:
        source = FakeStatcastSource({2024: []})
        adjuster = StatcastRateAdjuster(source)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert "statcast_blended" not in result[0].metadata


class TestEdgeCases:
    def test_empty_list(self) -> None:
        source = FakeStatcastSource({})
        adjuster = StatcastRateAdjuster(source)
        result = adjuster.adjust([])
        assert result == []

    def test_all_zero_rates(self) -> None:
        source = FakeStatcastSource({2024: [SAMPLE_STATCAST]})
        config = StatcastBlendConfig(blend_weight=0.5)
        adjuster = StatcastRateAdjuster(source, config)
        batter = _make_batter(
            rates={
                "hr": 0.0,
                "singles": 0.0,
                "doubles": 0.0,
                "triples": 0.0,
                "bb": 0.0,
                "so": 0.0,
                "hbp": 0.0,
                "sf": 0.0,
                "sh": 0.0,
            }
        )
        result = adjuster.adjust([batter])
        # Should not crash, all rates non-negative
        for rate in result[0].rates.values():
            assert rate >= 0

    def test_extreme_barrel_rate(self) -> None:
        extreme = StatcastBatterStats(
            player_id="mlb1",
            name="Aaron Judge",
            year=2024,
            pa=600,
            barrel_rate=0.30,
            hard_hit_rate=0.55,
            xwoba=0.430,
            xba=0.300,
            xslg=0.600,
        )
        source = FakeStatcastSource({2024: [extreme]})
        adjuster = StatcastRateAdjuster(source)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates["hr"] > 0
        assert result[0].rates["singles"] >= 0


class FakeEmptyStatcastSource:
    """Source that returns no data — used to verify FeatureStore delegation."""

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return []


class FakeFullStatcastSource:
    """Satisfies FullStatcastDataSource for FeatureStore construction."""

    def __init__(self, data: dict[int, list[StatcastBatterStats]]) -> None:
        self._data = data

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return self._data.get(year, [])

    def pitcher_expected_stats(self, year: int) -> list:
        return []


class FakeSkillDataSource:
    def batter_skill_stats(self, year: int) -> list:
        return []

    def pitcher_skill_stats(self, year: int) -> list:
        return []


class FakeBattedBallSource:
    def pitcher_batted_ball_stats(self, year: int) -> list:
        return []


class TestFeatureStoreIntegration:
    def test_uses_feature_store(self) -> None:
        """When feature_store is provided, the adjuster uses it instead of direct source."""
        # Direct source returns nothing — if it were used, blend wouldn't happen
        empty_source = FakeEmptyStatcastSource()
        # FeatureStore has actual data
        store = FeatureStore(
            statcast_source=FakeFullStatcastSource({2024: [SAMPLE_STATCAST]}),
            batted_ball_source=FakeBattedBallSource(),
            skill_data_source=FakeSkillDataSource(),
        )
        adjuster = StatcastRateAdjuster(
            statcast_source=empty_source, feature_store=store
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        # Since FeatureStore has data, blending should occur
        assert result[0].metadata.get("statcast_blended") is True
