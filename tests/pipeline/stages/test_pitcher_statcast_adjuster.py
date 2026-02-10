import pytest

from fantasy_baseball_manager.pipeline.stages.pitcher_statcast_adjuster import (
    PitcherStatcastAdjuster,
    PitcherStatcastConfig,
)
from fantasy_baseball_manager.pipeline.statcast_data import StatcastPitcherStats
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates
from fantasy_baseball_manager.player.identity import Player
from tests.conftest import make_test_feature_store


class FakePitcherStatcastSource:
    def __init__(self, data: dict[int, list[StatcastPitcherStats]]) -> None:
        self._data = data

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        return self._data.get(year, [])


def _make_pitcher(
    player_id: str = "fgp1",
    name: str = "Test Pitcher",
    rates: dict[str, float] | None = None,
    metadata: PlayerMetadata | None = None,
    mlbam_id: str | None = "mlb1",
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
    default_metadata: PlayerMetadata = {"is_starter": True, "ip_per_year": 180.0}
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
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id=mlbam_id, name=name),
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
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id="mlb1", name="Test Batter"),
    )


SAMPLE_PITCHER_STATCAST = StatcastPitcherStats(
    player_id="mlb1",
    name="Test Pitcher",
    year=2024,
    pa=700,
    xba=0.230,
    xslg=0.380,
    xwoba=0.300,
    xera=3.24,
    barrel_rate=0.07,
    hard_hit_rate=0.32,
)


class TestHitRateBlending:
    def test_weight_zero_returns_original(self) -> None:
        config = PitcherStatcastConfig(h_blend_weight=0.0, er_blend_weight=0.0)
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates["h"] == pytest.approx(pitcher.rates["h"])
        assert result[0].rates["er"] == pytest.approx(pitcher.rates["er"])

    def test_weight_one_returns_statcast_derived(self) -> None:
        config = PitcherStatcastConfig(h_blend_weight=1.0, er_blend_weight=1.0)
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # With weight=1.0, h and er should differ from original
        assert result[0].rates["h"] != pytest.approx(pitcher.rates["h"])
        assert result[0].rates["er"] != pytest.approx(pitcher.rates["er"])

    def test_partial_blend_interpolates(self) -> None:
        config = PitcherStatcastConfig(h_blend_weight=0.5, er_blend_weight=0.5)
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # Blended should be between original and statcast-derived
        original_h = pitcher.rates["h"]
        blended_h = result[0].rates["h"]
        assert blended_h != pytest.approx(original_h)


class TestIndependentWeights:
    def test_h_zero_er_full_leaves_h_unchanged(self) -> None:
        """h_blend_weight=0 keeps h at marcel; er_blend_weight=1 uses statcast er."""
        config = PitcherStatcastConfig(h_blend_weight=0.0, er_blend_weight=1.0)
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # h unchanged (weight=0)
        assert result[0].rates["h"] == pytest.approx(pitcher.rates["h"])
        # er fully statcast-derived (weight=1)
        expected_er = SAMPLE_PITCHER_STATCAST.xera / 27.0
        assert result[0].rates["er"] == pytest.approx(expected_er, abs=1e-6)

    def test_h_full_er_zero_leaves_er_unchanged(self) -> None:
        """h_blend_weight=1 uses statcast h; er_blend_weight=0 keeps er at marcel."""
        config = PitcherStatcastConfig(h_blend_weight=1.0, er_blend_weight=0.0)
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # h fully statcast-derived (weight=1)
        expected_h = SAMPLE_PITCHER_STATCAST.xba * (1.0 - 0.080 - 0.008)
        assert result[0].rates["h"] == pytest.approx(expected_h, abs=1e-6)
        # er unchanged (weight=0)
        assert result[0].rates["er"] == pytest.approx(pitcher.rates["er"])


class TestErDerivation:
    def test_er_derived_from_xera(self) -> None:
        """xERA / 27 gives ER per out."""
        config = PitcherStatcastConfig(h_blend_weight=1.0, er_blend_weight=1.0)
        statcast = StatcastPitcherStats(
            player_id="mlb1",
            name="Test",
            year=2024,
            pa=700,
            xba=0.230,
            xslg=0.380,
            xwoba=0.300,
            xera=2.70,  # exactly 0.10 ER per out
            barrel_rate=0.07,
            hard_hit_rate=0.32,
        )
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [statcast]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # xERA=2.70 => x_er = 2.70/27 = 0.10
        assert result[0].rates["er"] == pytest.approx(2.70 / 27.0, abs=1e-6)


class TestHitDerivation:
    def test_h_derived_from_xba_against(self) -> None:
        """x_h = xba * ab_per_bf where ab_per_bf = 1 - bb - hbp."""
        config = PitcherStatcastConfig(h_blend_weight=1.0, er_blend_weight=1.0)
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        # ab_per_bf = 1 - bb(0.080) - hbp(0.008) = 0.912
        # x_h = xba(0.230) * 0.912 = 0.20976
        expected_x_h = 0.230 * (1.0 - 0.080 - 0.008)
        assert result[0].rates["h"] == pytest.approx(expected_x_h, abs=1e-6)


class TestPassthrough:
    def test_batter_passes_through(self) -> None:
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_no_statcast_data_passes_through(self) -> None:
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: []}),
            ),
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates

    def test_low_pa_passes_through(self) -> None:
        low_pa = StatcastPitcherStats(
            player_id="mlb1",
            name="Low PA Pitcher",
            year=2024,
            pa=100,
            xba=0.230,
            xslg=0.380,
            xwoba=0.300,
            xera=3.50,
            barrel_rate=0.07,
            hard_hit_rate=0.32,
        )
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [low_pa]}),
            ),
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates

    def test_empty_list(self) -> None:
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({}),
            ),
        )
        result = adjuster.adjust([])
        assert result == []

    def test_missing_required_rates_passes_through(self) -> None:
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
        )
        # Pitcher with no h or er rates
        pitcher = _make_pitcher(rates={"so": 0.220, "bb": 0.080})
        # Remove h and er explicitly
        pitcher.rates.pop("h", None)
        pitcher.rates.pop("er", None)
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates


class TestNonBlendedRatesUnchanged:
    def test_bb_so_hbp_unchanged(self) -> None:
        config = PitcherStatcastConfig(h_blend_weight=0.5, er_blend_weight=0.5)
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
            config=config,
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates["bb"] == pitcher.rates["bb"]
        assert result[0].rates["so"] == pitcher.rates["so"]
        assert result[0].rates["hbp"] == pitcher.rates["hbp"]
        assert result[0].rates["hr"] == pitcher.rates["hr"]


class TestMetadata:
    def test_diagnostics_stored(self) -> None:
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].metadata["pitcher_xera"] == 3.24
        assert result[0].metadata["pitcher_xba_against"] == 0.230
        assert result[0].metadata["pitcher_statcast_blended"] is True
        assert "pitcher_h_blend_weight" in result[0].metadata
        assert "pitcher_er_blend_weight" in result[0].metadata

    def test_existing_metadata_preserved(self) -> None:
        adjuster = PitcherStatcastAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakePitcherStatcastSource({2024: [SAMPLE_PITCHER_STATCAST]}),
            ),
        )
        pitcher = _make_pitcher(metadata={"custom_key": "custom_value"})  # type: ignore[typeddict-unknown-key]
        result = adjuster.adjust([pitcher])
        assert result[0].metadata["custom_key"] == "custom_value"  # type: ignore[typeddict-item]
        assert result[0].metadata["pitcher_statcast_blended"] is True
