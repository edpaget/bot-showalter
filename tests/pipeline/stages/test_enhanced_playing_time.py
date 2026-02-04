import pytest

from fantasy_baseball_manager.pipeline.stages.enhanced_playing_time import (
    EnhancedPlayingTimeProjector,
)
from fantasy_baseball_manager.pipeline.stages.playing_time_config import (
    PlayingTimeConfig,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates


class TestInjuryProxy:
    def test_healthy_batter_no_penalty(self) -> None:
        """Healthy player with 162 games gets injury_factor = 1.0."""
        p = PlayerRates(
            player_id="p1",
            name="Healthy Batter",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [650, 600, 580],
                "games_per_year": [162.0, 160.0, 158.0],
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert result[0].metadata["injury_factor"] == pytest.approx(1.0)

    def test_injured_batter_penalty(self) -> None:
        """Batter with 100 games should have injury_factor < 1.0 but > min."""
        p = PlayerRates(
            player_id="p1",
            name="Injured Batter",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [400, 600, 580],
                "games_per_year": [100.0, 150.0, 145.0],
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        injury_factor = result[0].metadata["injury_factor"]
        assert isinstance(injury_factor, float)
        assert injury_factor < 1.0
        assert injury_factor > 0.70  # Not at minimum (100/162 = 61.7% > 50%)

    def test_severely_injured_batter_max_penalty(self) -> None:
        """Batter with 40 games (below min_games_pct) gets max penalty."""
        p = PlayerRates(
            player_id="p1",
            name="Severely Injured",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [160, 600, 580],
                "games_per_year": [40.0, 150.0, 145.0],
            },
        )
        config = PlayingTimeConfig(games_played_weight=0.30, min_games_pct=0.50)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        injury_factor = result[0].metadata["injury_factor"]
        assert isinstance(injury_factor, float)
        # At minimum: 1.0 - 0.30 = 0.70
        assert injury_factor == pytest.approx(0.70)

    def test_healthy_starter_no_penalty(self) -> None:
        """Healthy starter with 30+ starts gets injury_factor ≈ 1.0."""
        p = PlayerRates(
            player_id="sp1",
            name="Healthy Starter",
            year=2025,
            age=28,
            rates={"so": 0.25},
            metadata={
                "ip_per_year": [180.0, 170.0, 165.0],
                "games_per_year": [32.0, 30.0, 28.0],
                "is_starter": True,
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert result[0].metadata["injury_factor"] == pytest.approx(1.0)

    def test_injured_starter_penalty(self) -> None:
        """Starter with only 15 starts should have penalty."""
        p = PlayerRates(
            player_id="sp1",
            name="Injured Starter",
            year=2025,
            age=28,
            rates={"so": 0.25},
            metadata={
                "ip_per_year": [90.0, 170.0, 165.0],
                "games_per_year": [15.0, 30.0, 28.0],
                "is_starter": True,
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        injury_factor = result[0].metadata["injury_factor"]
        assert isinstance(injury_factor, float)
        assert injury_factor < 1.0


class TestAgeFactor:
    def test_age_28_no_decline(self) -> None:
        """Age 28 (below threshold) should have no decline."""
        p = PlayerRates(
            player_id="p1",
            name="Young Player",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [600, 580, 560],
                "games_per_year": [150.0, 145.0, 140.0],
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert result[0].metadata["age_pt_factor"] == pytest.approx(1.0)

    def test_age_35_nine_percent_decline(self) -> None:
        """Age 35 (3 years past 32) should have 9% decline."""
        p = PlayerRates(
            player_id="p1",
            name="Veteran",
            year=2025,
            age=35,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [600, 580, 560],
                "games_per_year": [150.0, 145.0, 140.0],
            },
        )
        config = PlayingTimeConfig(age_decline_start=32, age_decline_rate=0.03)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        # 3 years * 3% = 9% decline
        assert result[0].metadata["age_pt_factor"] == pytest.approx(0.91)

    def test_age_40_significant_decline(self) -> None:
        """Age 40 (8 years past 32) should have 24% decline."""
        p = PlayerRates(
            player_id="p1",
            name="Old Player",
            year=2025,
            age=40,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [500, 480, 460],
                "games_per_year": [125.0, 120.0, 115.0],
            },
        )
        config = PlayingTimeConfig(age_decline_start=32, age_decline_rate=0.03)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        # 8 years * 3% = 24% decline
        assert result[0].metadata["age_pt_factor"] == pytest.approx(0.76)


class TestVolatility:
    def test_stable_pa_no_penalty(self) -> None:
        """Stable PA (600, 580, 610) should have no volatility penalty."""
        p = PlayerRates(
            player_id="p1",
            name="Stable Player",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [600, 580, 610],
                "games_per_year": [150.0, 145.0, 152.0],
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert result[0].metadata["volatility_factor"] == pytest.approx(1.0)

    def test_volatile_pa_penalty(self) -> None:
        """Volatile PA (600, 300, 550) should have penalty applied."""
        p = PlayerRates(
            player_id="p1",
            name="Volatile Player",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [600, 300, 550],
                "games_per_year": [150.0, 75.0, 137.0],
            },
        )
        config = PlayingTimeConfig(volatility_threshold=0.25, volatility_penalty=0.10)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        # 600 -> 300 is a 50% drop, well above 25% threshold
        assert result[0].metadata["volatility_factor"] == pytest.approx(0.90)

    def test_single_year_no_volatility(self) -> None:
        """Single year of data should have no volatility penalty."""
        p = PlayerRates(
            player_id="p1",
            name="Rookie",
            year=2025,
            age=23,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [400],
                "games_per_year": [100.0],
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert result[0].metadata["volatility_factor"] == pytest.approx(1.0)


class TestRoleCaps:
    def test_batter_pa_capped_at_700(self) -> None:
        """Base projection of 750 PA should be capped at 700."""
        # To get base PA > 700, we need high historical PA
        # Marcel formula: 0.5 * PA_y1 + 0.1 * PA_y2 + 200
        # With PA_y1=700, PA_y2=700: 0.5*700 + 0.1*700 + 200 = 620
        # With PA_y1=900, PA_y2=900: 0.5*900 + 0.1*900 + 200 = 740
        p = PlayerRates(
            player_id="p1",
            name="Heavy Usage",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [900, 900, 850],
                "games_per_year": [225.0, 225.0, 212.0],  # Unrealistic but tests cap
            },
        )
        config = PlayingTimeConfig(batter_pa_cap=700)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        assert result[0].opportunities == 700

    def test_starter_ip_capped_at_200(self) -> None:
        """Starter with base projection > 200 IP should be capped."""
        # Marcel: 0.5 * IP_y1 + 0.1 * IP_y2 + 60 (starter)
        # With IP_y1=300, IP_y2=300: 0.5*300 + 0.1*300 + 60 = 240
        p = PlayerRates(
            player_id="sp1",
            name="Workhorse",
            year=2025,
            age=28,
            rates={"so": 0.25},
            metadata={
                "ip_per_year": [300.0, 300.0, 280.0],
                "games_per_year": [50.0, 50.0, 47.0],
                "is_starter": True,
            },
        )
        config = PlayingTimeConfig(starter_ip_cap=200)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        # Opportunities are in outs (IP * 3)
        assert result[0].opportunities == pytest.approx(600.0)  # 200 IP * 3

    def test_reliever_ip_capped_at_80(self) -> None:
        """Reliever with base projection > 80 IP should be capped."""
        # Marcel: 0.5 * IP_y1 + 0.1 * IP_y2 + 25 (reliever)
        # With IP_y1=100, IP_y2=100: 0.5*100 + 0.1*100 + 25 = 85
        p = PlayerRates(
            player_id="rp1",
            name="Heavy Reliever",
            year=2025,
            age=28,
            rates={"so": 0.30},
            metadata={
                "ip_per_year": [100.0, 100.0, 95.0],
                "games_per_year": [100.0, 100.0, 95.0],
                "is_starter": False,
            },
        )
        config = PlayingTimeConfig(reliever_ip_cap=80)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        # Opportunities are in outs (IP * 3)
        assert result[0].opportunities == pytest.approx(240.0)  # 80 IP * 3

    def test_catcher_pa_capped_at_550(self) -> None:
        """Catcher projection should be capped at 550 PA."""
        p = PlayerRates(
            player_id="c1",
            name="Starting Catcher",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [700, 650, 620],
                "games_per_year": [175.0, 162.0, 155.0],
                "position": "C",
            },
        )
        config = PlayingTimeConfig(catcher_pa_cap=550, batter_pa_cap=700)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        assert result[0].opportunities == 550


class TestPassthrough:
    def test_missing_metadata_falls_back_to_opportunities(self) -> None:
        """Player without pa_per_year or ip_per_year uses existing opportunities."""
        p = PlayerRates(
            player_id="p1",
            name="Unknown",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            opportunities=500.0,
            metadata={},
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert result[0].opportunities == 500.0

    def test_existing_metadata_preserved(self) -> None:
        """Existing metadata should be preserved after projection."""
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={  # type: ignore[invalid-argument-type]
                "pa_per_year": [600.0, 580.0],
                "games_per_year": [150.0, 145.0],
                "team": "NYY",
                "custom_field": "custom_value",  # type: ignore[invalid-key]
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert result[0].metadata["team"] == "NYY"
        assert result[0].metadata["custom_field"] == "custom_value"  # type: ignore[typeddict-item]

    def test_pt_metadata_added(self) -> None:
        """Playing time diagnostic metadata should be added."""
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [600, 580],
                "games_per_year": [150.0, 145.0],
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert "injury_factor" in result[0].metadata
        assert "age_pt_factor" in result[0].metadata
        assert "volatility_factor" in result[0].metadata
        assert "base_pa" in result[0].metadata


class TestIntegration:
    def test_all_factors_combined(self) -> None:
        """Test that all factors are applied multiplicatively."""
        # Age 35: 0.91 age factor (3 years * 3%)
        # Healthy: 1.0 injury factor
        # Stable: 1.0 volatility factor
        # Base PA: 0.5 * 600 + 0.1 * 580 + 200 = 558
        # Expected: 558 * 0.91 * 1.0 * 1.0 = 507.78
        p = PlayerRates(
            player_id="p1",
            name="Veteran Healthy",
            year=2025,
            age=35,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [600, 580],
                "games_per_year": [162.0, 160.0],
            },
        )
        config = PlayingTimeConfig(age_decline_start=32, age_decline_rate=0.03)
        projector = EnhancedPlayingTimeProjector(config=config)
        result = projector.project([p])

        expected = 558 * 0.91
        assert result[0].opportunities == pytest.approx(expected, rel=0.01)

    def test_inferred_games_from_pa(self) -> None:
        """Test that games are inferred from PA when games_per_year not provided."""
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=28,
            rates={"hr": 0.04},
            metadata={
                "pa_per_year": [648, 580],  # 648 / 4 = 162 games
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        # Should not get an injury penalty since 162 games is full season
        assert result[0].metadata["injury_factor"] == pytest.approx(1.0)

    def test_pitcher_metadata_includes_base_ip(self) -> None:
        """Pitcher projection should include base_ip in metadata."""
        p = PlayerRates(
            player_id="sp1",
            name="Starter",
            year=2025,
            age=28,
            rates={"so": 0.25},
            metadata={
                "ip_per_year": [180.0, 170.0],
                "games_per_year": [32.0, 30.0],
                "is_starter": True,
            },
        )
        projector = EnhancedPlayingTimeProjector()
        result = projector.project([p])

        assert "base_ip" in result[0].metadata
        # Marcel: 0.5 * 180 + 0.1 * 170 + 60 = 167
        assert result[0].metadata["base_ip"] == pytest.approx(167.0)
