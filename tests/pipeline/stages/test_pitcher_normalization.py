import pytest

from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationAdjuster,
    PitcherNormalizationConfig,
)
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates


def _make_pitcher(
    rates: dict[str, float] | None = None,
    metadata: PlayerMetadata | None = None,
) -> PlayerRates:
    default_rates = {
        "h": 0.24,
        "hr": 0.03,
        "bb": 0.08,
        "hbp": 0.01,
        "so": 0.22,
        "er": 0.12,
    }
    default_metadata: PlayerMetadata = {
        "ip_per_year": 180.0,
        "is_starter": True,
    }
    return PlayerRates(
        player_id="p1",
        name="Test Pitcher",
        year=2025,
        age=28,
        rates=rates or default_rates,
        opportunities=600.0,
        metadata=metadata if metadata is not None else default_metadata,
    )


def _make_batter(
    rates: dict[str, float] | None = None,
) -> PlayerRates:
    return PlayerRates(
        player_id="b1",
        name="Test Batter",
        year=2025,
        age=28,
        rates=rates or {"h": 0.25, "hr": 0.04, "bb": 0.09, "so": 0.20},
        opportunities=600.0,
        metadata={"pa_per_year": [550.0]},
    )


class TestBatterPassthrough:
    def test_batter_without_pitcher_metadata_unchanged(self) -> None:
        """Batters (no ip_per_year or is_starter) pass through unchanged."""
        adjuster = PitcherNormalizationAdjuster()
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_mixed_batters_and_pitchers(self) -> None:
        """Only pitchers are adjusted when mixed list is provided."""
        adjuster = PitcherNormalizationAdjuster()
        batter = _make_batter()
        pitcher = _make_pitcher()
        result = adjuster.adjust([batter, pitcher])
        assert result[0].rates == batter.rates
        assert result[1].rates != pitcher.rates  # pitcher should be adjusted


class TestBabipComputation:
    def test_babip_from_known_rates(self) -> None:
        """BABIP = (h - hr) / (1 + h - hr - so) for per-out rates."""
        # h=0.24, hr=0.03, so=0.22
        # babip = (0.24 - 0.03) / (1 + 0.24 - 0.03 - 0.22) = 0.21 / 0.99
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        expected_babip = 0.21 / 0.99
        assert result[0].metadata["observed_babip"] == pytest.approx(expected_babip, abs=1e-6)

    def test_babip_at_league_mean_no_h_adjustment(self) -> None:
        """When observed BABIP equals league mean, H rate stays the same."""
        # Set rates so observed BABIP = 0.300 exactly
        # babip = (h - hr) / (1 + h - hr - so) = 0.300
        # With hr=0.03, so=0.22: (h - 0.03) / (1 + h - 0.03 - 0.22) = 0.300
        # (h - 0.03) = 0.300 * (0.75 + h) = 0.225 + 0.300*h
        # h - 0.300*h = 0.225 + 0.03 = 0.255
        # 0.700*h = 0.255
        # h = 0.255 / 0.700 ≈ 0.36428...
        h_at_league = 0.255 / 0.700
        rates = {"h": h_at_league, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.12}
        config = PitcherNormalizationConfig(babip_regression_weight=0.5)
        adjuster = PitcherNormalizationAdjuster(config)
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert result[0].rates["h"] == pytest.approx(h_at_league, abs=1e-6)


class TestBabipRegression:
    def test_high_babip_regresses_downward(self) -> None:
        """Pitcher with high BABIP should have H rate decreased."""
        # h=0.40 gives high BABIP
        rates = {"h": 0.40, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.15}
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert result[0].rates["h"] < 0.40

    def test_low_babip_regresses_upward(self) -> None:
        """Pitcher with low BABIP should have H rate increased."""
        # h=0.15 gives low BABIP
        rates = {"h": 0.15, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.08}
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert result[0].rates["h"] > 0.15

    def test_h_rate_round_trip(self) -> None:
        """With regression weight 0, H rate should not change."""
        rates = {"h": 0.24, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.12}
        config = PitcherNormalizationConfig(babip_regression_weight=0.0)
        adjuster = PitcherNormalizationAdjuster(config)
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert result[0].rates["h"] == pytest.approx(0.24, abs=1e-6)


class TestLobComputation:
    def test_high_k_pitcher_higher_expected_lob(self) -> None:
        """A high-K pitcher should get a higher expected LOB%."""
        # k_pct well above league average -> expected LOB > baseline
        rates = {"h": 0.24, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.35, "er": 0.12}
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert float(result[0].metadata["expected_lob_pct"]) > 0.73  
    def test_low_k_pitcher_lower_expected_lob(self) -> None:
        """A low-K pitcher should get a lower expected LOB%."""
        rates = {"h": 0.24, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.10, "er": 0.12}
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert float(result[0].metadata["expected_lob_pct"]) < 0.73  
    def test_lob_clamped_to_min(self) -> None:
        """LOB% should not go below min_lob."""
        # Very low K rate to push LOB below min
        rates = {"h": 0.24, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.01, "er": 0.12}
        config = PitcherNormalizationConfig(min_lob=0.65)
        adjuster = PitcherNormalizationAdjuster(config)
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert float(result[0].metadata["expected_lob_pct"]) >= 0.65  
    def test_lob_clamped_to_max(self) -> None:
        """LOB% should not exceed max_lob."""
        # Very high K rate to push LOB above max
        rates = {"h": 0.24, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.90, "er": 0.12}
        config = PitcherNormalizationConfig(max_lob=0.82)
        adjuster = PitcherNormalizationAdjuster(config)
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert float(result[0].metadata["expected_lob_pct"]) <= 0.82  

class TestErAdjustment:
    def test_end_to_end_er_adjustment(self) -> None:
        """ER rate should be a blend of expected and observed."""
        rates = {"h": 0.40, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.20}
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        # ER should change from original
        assert result[0].rates["er"] != pytest.approx(0.20, abs=1e-4)

    def test_other_rates_preserved(self) -> None:
        """Rates other than h and er should be unchanged."""
        rates = {"h": 0.30, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.12}
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert result[0].rates["hr"] == pytest.approx(0.03)
        assert result[0].rates["bb"] == pytest.approx(0.08)
        assert result[0].rates["hbp"] == pytest.approx(0.01)
        assert result[0].rates["so"] == pytest.approx(0.22)


class TestMetadataAndImmutability:
    def test_metadata_preserved_and_diagnostics_added(self) -> None:
        """Original metadata should be preserved, diagnostics added."""
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher()
        result = adjuster.adjust([pitcher])
        assert result[0].metadata["ip_per_year"] == 180.0
        assert result[0].metadata["is_starter"] is True
        assert "observed_babip" in result[0].metadata
        assert "expected_babip" in result[0].metadata
        assert "expected_lob_pct" in result[0].metadata

    def test_original_player_rates_not_mutated(self) -> None:
        """The input PlayerRates should not be modified."""
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher()
        original_rates = dict(pitcher.rates)
        original_metadata = dict(pitcher.metadata)
        adjuster.adjust([pitcher])
        assert pitcher.rates == original_rates
        assert pitcher.metadata == original_metadata


class TestCustomConfig:
    def test_custom_config_overrides_behavior(self) -> None:
        """Custom config values should change the adjustment."""
        rates = {"h": 0.30, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.12}
        default_adjuster = PitcherNormalizationAdjuster()
        custom_adjuster = PitcherNormalizationAdjuster(
            PitcherNormalizationConfig(
                league_babip=0.280,
                babip_regression_weight=0.8,
                lob_regression_weight=0.9,
            )
        )
        pitcher1 = _make_pitcher(rates=dict(rates))
        pitcher2 = _make_pitcher(rates=dict(rates))
        result_default = default_adjuster.adjust([pitcher1])
        result_custom = custom_adjuster.adjust([pitcher2])
        assert result_default[0].rates["h"] != pytest.approx(result_custom[0].rates["h"], abs=1e-6)


class TestEdgeCases:
    def test_missing_rate_keys_passthrough(self) -> None:
        """Pitcher without required rate keys passes through unchanged."""
        rates = {"bb": 0.08}  # missing h, hr, so, er
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        assert result[0].rates == rates

    def test_extreme_so_rate_safe(self) -> None:
        """Very high SO rate (tiny BIP denominator) should not crash."""
        # so=0.95 means BIP denominator = 1 + h - hr - so is very small
        rates = {"h": 0.04, "hr": 0.01, "bb": 0.05, "hbp": 0.01, "so": 0.95, "er": 0.05}
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates)
        result = adjuster.adjust([pitcher])
        # Should not crash, rates should be finite
        for val in result[0].rates.values():
            assert val == val  # not NaN
            assert abs(val) < 100  # not extreme

    def test_league_rates_from_metadata(self) -> None:
        """When avg_league_rates has BABIP-relevant keys, use them."""
        rates = {"h": 0.30, "hr": 0.03, "bb": 0.08, "hbp": 0.01, "so": 0.22, "er": 0.12}
        metadata: PlayerMetadata = {
            "ip_per_year": 180.0,
            "is_starter": True,
            "avg_league_rates": {"h": 0.25, "hr": 0.03, "so": 0.22},
        }
        adjuster = PitcherNormalizationAdjuster()
        pitcher = _make_pitcher(rates=rates, metadata=metadata)
        result = adjuster.adjust([pitcher])
        # Should use the league rates to derive league BABIP
        assert "expected_babip" in result[0].metadata
