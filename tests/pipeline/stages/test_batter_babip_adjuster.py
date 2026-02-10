import pytest

from fantasy_baseball_manager.pipeline.stages.batter_babip_adjuster import (
    BatterBabipAdjuster,
    BatterBabipConfig,
)
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates
from fantasy_baseball_manager.player.identity import Player
from tests.conftest import make_test_feature_store


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


class TestXBabipDerivation:
    """Verify expected BABIP math from xBA and barrel_rate."""

    def test_x_babip_basic_computation(self) -> None:
        """x_babip = (xba * ab_per_pa - xHR_rate) / bip_rate."""
        config = BatterBabipConfig(adjustment_weight=1.0)
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [SAMPLE_STATCAST]}),
            ),
            config=config,
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])

        # Compute expected values manually:
        # bb=0.08, so=0.20, hbp=0.01, sf=0.005, sh=0.0
        # bip_rate = 1 - (0.08 + 0.20 + 0.01 + 0.005 + 0.0) = 0.705
        # ab_per_pa = 1 - (0.08 + 0.01 + 0.005 + 0.0) = 0.905
        # xHR_rate = 0.15 * 0.55 * 0.705 = 0.0581625
        # x_babip = (0.270 * 0.905 - 0.0581625) / 0.705
        bip_rate = 0.705
        ab_per_pa = 0.905
        x_hr = 0.15 * 0.55 * bip_rate
        x_babip = (0.270 * ab_per_pa - x_hr) / bip_rate

        assert result[0].metadata["x_babip"] == pytest.approx(x_babip, abs=1e-6)

    def test_high_barrel_rate_lowers_x_babip(self) -> None:
        """Higher barrel rate -> more xHR -> lower xBABIP (more hits are HR)."""
        high_barrel = StatcastBatterStats(
            player_id="mlb1",
            name="Power Hitter",
            year=2024,
            pa=500,
            barrel_rate=0.30,
            hard_hit_rate=0.55,
            xwoba=0.400,
            xba=0.270,
            xslg=0.550,
        )
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [high_barrel]}),
            ),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])

        # Compare with normal barrel rate
        adjuster2 = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [SAMPLE_STATCAST]}),
            ),
        )
        result2 = adjuster2.adjust([batter])

        assert result[0].metadata["x_babip"] < result2[0].metadata["x_babip"]


class TestSinglesAdjustment:
    def test_above_average_xbabip_increases_singles(self) -> None:
        # Use xBA that will produce a high xBABIP
        high_xba = StatcastBatterStats(
            player_id="mlb1",
            name="High BABIP",
            year=2024,
            pa=500,
            barrel_rate=0.05,  # low barrel rate
            hard_hit_rate=0.35,
            xwoba=0.340,
            xba=0.310,  # high xBA
            xslg=0.430,
        )
        config = BatterBabipConfig(adjustment_weight=0.5)
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [high_xba]}),
            ),
            config=config,
        )
        # Batter with low singles rate -> observed BABIP will be low
        batter = _make_batter(rates={"singles": 0.100})
        result = adjuster.adjust([batter])
        assert result[0].rates["singles"] > 0.100

    def test_below_average_xbabip_decreases_singles(self) -> None:
        low_xba = StatcastBatterStats(
            player_id="mlb1",
            name="Low BABIP",
            year=2024,
            pa=500,
            barrel_rate=0.15,
            hard_hit_rate=0.42,
            xwoba=0.300,
            xba=0.200,  # low xBA
            xslg=0.380,
        )
        config = BatterBabipConfig(adjustment_weight=0.5)
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [low_xba]}),
            ),
            config=config,
        )
        # Batter with high singles rate -> observed BABIP will be high
        batter = _make_batter(rates={"singles": 0.180})
        result = adjuster.adjust([batter])
        assert result[0].rates["singles"] < 0.180

    def test_weight_zero_no_change(self) -> None:
        config = BatterBabipConfig(adjustment_weight=0.0)
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [SAMPLE_STATCAST]}),
            ),
            config=config,
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates["singles"] == pytest.approx(batter.rates["singles"])

    def test_singles_cannot_go_negative(self) -> None:
        # Very low xBA to force a large negative adjustment
        very_low = StatcastBatterStats(
            player_id="mlb1",
            name="Bad Hitter",
            year=2024,
            pa=500,
            barrel_rate=0.20,
            hard_hit_rate=0.30,
            xwoba=0.200,
            xba=0.100,  # extremely low
            xslg=0.200,
        )
        config = BatterBabipConfig(adjustment_weight=1.0)
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [very_low]}),
            ),
            config=config,
        )
        batter = _make_batter(rates={"singles": 0.010})
        result = adjuster.adjust([batter])
        assert result[0].rates["singles"] >= 0.0


class TestPassthrough:
    def test_pitcher_passes_through(self) -> None:
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [SAMPLE_STATCAST]}),
            ),
        )
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].rates == pitcher.rates

    def test_no_statcast_data_passes_through(self) -> None:
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: []}),
            ),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_low_pa_passes_through(self) -> None:
        low_pa = StatcastBatterStats(
            player_id="mlb1",
            name="Low PA",
            year=2024,
            pa=50,
            barrel_rate=0.15,
            hard_hit_rate=0.42,
            xwoba=0.360,
            xba=0.270,
            xslg=0.480,
        )
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [low_pa]}),
            ),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_empty_list(self) -> None:
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({}),
            ),
        )
        result = adjuster.adjust([])
        assert result == []

    def test_zero_bip_rate_passes_through(self) -> None:
        """When all PAs are walks/strikeouts/HBP, BIP rate is zero."""
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [SAMPLE_STATCAST]}),
            ),
        )
        batter = _make_batter(
            rates={
                "hr": 0.0,
                "singles": 0.0,
                "doubles": 0.0,
                "triples": 0.0,
                "bb": 0.500,
                "so": 0.400,
                "hbp": 0.050,
                "sf": 0.050,
                "sh": 0.000,
            }
        )
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates


class TestMetadata:
    def test_diagnostics_stored(self) -> None:
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [SAMPLE_STATCAST]}),
            ),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert "x_babip" in result[0].metadata
        assert "observed_batter_babip" in result[0].metadata
        assert "babip_singles_adjustment" in result[0].metadata

    def test_existing_metadata_preserved(self) -> None:
        adjuster = BatterBabipAdjuster(
            feature_store=make_test_feature_store(
                statcast_source=FakeStatcastSource({2024: [SAMPLE_STATCAST]}),
            ),
        )
        batter = _make_batter(metadata={"statcast_blended": True, "custom_key": 42})  # type: ignore[typeddict-unknown-key]
        result = adjuster.adjust([batter])
        assert result[0].metadata["statcast_blended"] is True
        assert result[0].metadata["custom_key"] == 42  # type: ignore[typeddict-item]
        assert "x_babip" in result[0].metadata
