"""Tests for ML feature extraction."""

import pytest

from fantasy_baseball_manager.ml.features import (
    LEAGUE_AVG_CHASE_RATE,
    LEAGUE_AVG_WHIFF_RATE,
    BatterFeatureExtractor,
    PitcherFeatureExtractor,
)
from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats, StatcastPitcherStats
from fantasy_baseball_manager.pipeline.types import PlayerRates


class TestBatterFeatureExtractor:
    def test_feature_names_returns_expected_list(self) -> None:
        extractor = BatterFeatureExtractor()
        names = extractor.feature_names()

        assert "marcel_hr" in names
        assert "xba" in names
        assert "barrel_rate" in names
        assert "age" in names
        assert "opportunities" in names
        assert len(names) == 25  # 18 original + 7 swing decision features

    def test_feature_names_includes_swing_decision_features(self) -> None:
        """Verify all swing decision features are present."""
        extractor = BatterFeatureExtractor()
        names = extractor.feature_names()

        swing_features = [
            "chase_rate",
            "whiff_rate",
            "chase_minus_league_avg",
            "whiff_minus_league_avg",
            "chase_x_whiff",
            "discipline_score",
            "has_skill_data",
        ]
        for feature in swing_features:
            assert feature in names, f"Missing feature: {feature}"

    def test_extract_returns_features_for_valid_player(self) -> None:
        extractor = BatterFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Batter",
            year=2024,
            age=28,
            rates={
                "hr": 0.035,
                "so": 0.200,
                "bb": 0.100,
                "singles": 0.150,
                "doubles": 0.050,
                "triples": 0.005,
                "sb": 0.020,
            },
            opportunities=500.0,
        )

        statcast = StatcastBatterStats(
            player_id="99999",
            name="Test Batter",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        features = extractor.extract(player, statcast)

        assert features is not None
        assert features.shape == (25,)
        assert features[0] == 0.035  # marcel_hr
        assert features[7] == 0.280  # xba
        assert features[19] == 28  # age (shifted by 7 new features)
        assert features[24] == 500.0  # opportunities (shifted)

    def test_extract_with_skill_data(self) -> None:
        """Test that skill data values are used when provided."""
        extractor = BatterFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Batter",
            year=2024,
            age=28,
            rates={
                "hr": 0.035,
                "so": 0.200,
                "bb": 0.100,
                "singles": 0.150,
                "doubles": 0.050,
                "triples": 0.005,
                "sb": 0.020,
            },
            opportunities=500.0,
        )

        statcast = StatcastBatterStats(
            player_id="99999",
            name="Test Batter",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        skill_data = BatterSkillStats(
            player_id="12345",
            name="Test Batter",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            exit_velo_avg=90.0,
            exit_velo_max=110.0,
            chase_rate=0.25,  # Below league avg (0.30)
            whiff_rate=0.08,  # Below league avg (0.105)
            sprint_speed=28.0,
        )

        features = extractor.extract(player, statcast, skill_data)

        assert features is not None
        # Check swing decision features (indices 12-18)
        assert features[12] == 0.25  # chase_rate
        assert features[13] == 0.08  # whiff_rate
        assert features[14] == pytest.approx(-0.05)  # chase_minus_league_avg (0.25 - 0.30)
        assert features[15] == pytest.approx(-0.025)  # whiff_minus_league_avg (0.08 - 0.105)
        assert features[16] == pytest.approx(0.25 * 0.08)  # chase_x_whiff
        assert features[17] == pytest.approx((1 - 0.25) * (1 - 0.08))  # discipline_score
        assert features[18] == 1.0  # has_skill_data

    def test_extract_without_skill_data_uses_league_averages(self) -> None:
        """Test that league averages are used when skill_data is None."""
        extractor = BatterFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Batter",
            year=2024,
            age=28,
            rates={
                "hr": 0.035,
                "so": 0.200,
                "bb": 0.100,
                "singles": 0.150,
                "doubles": 0.050,
                "triples": 0.005,
                "sb": 0.020,
            },
            opportunities=500.0,
        )

        statcast = StatcastBatterStats(
            player_id="99999",
            name="Test Batter",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        features = extractor.extract(player, statcast, skill_data=None)

        assert features is not None
        # Check swing decision features use league averages
        assert features[12] == LEAGUE_AVG_CHASE_RATE  # chase_rate
        assert features[13] == LEAGUE_AVG_WHIFF_RATE  # whiff_rate
        assert features[14] == 0.0  # chase_minus_league_avg (0.30 - 0.30)
        assert features[15] == 0.0  # whiff_minus_league_avg (0.105 - 0.105)
        assert features[18] == 0.0  # has_skill_data = False

    def test_extract_returns_none_below_min_pa(self) -> None:
        extractor = BatterFeatureExtractor(min_pa=100)

        player = PlayerRates(
            player_id="12345",
            name="Test Batter",
            year=2024,
            age=28,
            rates={
                "hr": 0.035,
                "so": 0.200,
                "bb": 0.100,
                "singles": 0.150,
                "doubles": 0.050,
                "triples": 0.005,
            },
            opportunities=500.0,
        )

        statcast = StatcastBatterStats(
            player_id="99999",
            name="Test Batter",
            year=2023,
            pa=50,  # Below min_pa
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        features = extractor.extract(player, statcast)
        assert features is None

    def test_extract_returns_none_missing_rates(self) -> None:
        extractor = BatterFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Batter",
            year=2024,
            age=28,
            rates={"hr": 0.035},  # Missing other required rates
            opportunities=500.0,
        )

        statcast = StatcastBatterStats(
            player_id="99999",
            name="Test Batter",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        features = extractor.extract(player, statcast)
        assert features is None

    def test_extract_handles_zero_hr_rate(self) -> None:
        """Test barrel_vs_hr_ratio handles zero HR rate."""
        extractor = BatterFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Batter",
            year=2024,
            age=28,
            rates={
                "hr": 0.0,  # Zero HR rate
                "so": 0.200,
                "bb": 0.100,
                "singles": 0.150,
                "doubles": 0.050,
                "triples": 0.005,
            },
            opportunities=500.0,
        )

        statcast = StatcastBatterStats(
            player_id="99999",
            name="Test Batter",
            year=2023,
            pa=450,
            barrel_rate=0.08,
            hard_hit_rate=0.40,
            xwoba=0.350,
            xba=0.280,
            xslg=0.450,
        )

        features = extractor.extract(player, statcast)

        assert features is not None
        # barrel_vs_hr_ratio should be 0 when HR rate is 0 (index 23 now)
        assert features[23] == 0.0


class TestPitcherFeatureExtractor:
    def test_feature_names_returns_expected_list(self) -> None:
        extractor = PitcherFeatureExtractor()
        names = extractor.feature_names()

        assert "marcel_h" in names
        assert "xera" in names
        assert "gb_pct" in names
        assert "age" in names
        assert "is_starter" in names
        assert len(names) == 21

    def test_extract_returns_features_with_batted_ball(self) -> None:
        extractor = PitcherFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Pitcher",
            year=2024,
            age=30,
            rates={
                "h": 0.080,
                "er": 0.030,
                "so": 0.090,
                "bb": 0.030,
                "hr": 0.010,
            },
            opportunities=600.0,
            metadata={"is_starter": True},
        )

        statcast = StatcastPitcherStats(
            player_id="99999",
            name="Test Pitcher",
            year=2023,
            pa=400,
            xba=0.240,
            xslg=0.380,
            xwoba=0.310,
            xera=3.50,
            barrel_rate=0.06,
            hard_hit_rate=0.35,
        )

        batted_ball = PitcherBattedBallStats(
            player_id="12345",
            name="Test Pitcher",
            year=2023,
            pa=400,
            gb_pct=0.48,
            fb_pct=0.32,
            ld_pct=0.18,
            iffb_pct=0.12,
        )

        features = extractor.extract(player, statcast, batted_ball)

        assert features is not None
        assert features.shape == (21,)
        assert features[0] == 0.080  # marcel_h
        assert features[8] == 3.50  # xera
        assert features[11] == 0.48  # gb_pct
        assert features[15] == 30  # age
        assert features[20] == 1.0  # is_starter

    def test_extract_uses_defaults_without_batted_ball(self) -> None:
        extractor = PitcherFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Pitcher",
            year=2024,
            age=30,
            rates={
                "h": 0.080,
                "er": 0.030,
                "so": 0.090,
                "bb": 0.030,
                "hr": 0.010,
            },
            opportunities=600.0,
        )

        statcast = StatcastPitcherStats(
            player_id="99999",
            name="Test Pitcher",
            year=2023,
            pa=400,
            xba=0.240,
            xslg=0.380,
            xwoba=0.310,
            xera=3.50,
            barrel_rate=0.06,
            hard_hit_rate=0.35,
        )

        features = extractor.extract(player, statcast, None)

        assert features is not None
        # Check defaults are used
        assert features[11] == 0.43  # gb_pct default
        assert features[12] == 0.35  # fb_pct default

    def test_extract_returns_none_below_min_pa(self) -> None:
        extractor = PitcherFeatureExtractor(min_pa=100)

        player = PlayerRates(
            player_id="12345",
            name="Test Pitcher",
            year=2024,
            age=30,
            rates={
                "h": 0.080,
                "er": 0.030,
                "so": 0.090,
                "bb": 0.030,
                "hr": 0.010,
            },
            opportunities=600.0,
        )

        statcast = StatcastPitcherStats(
            player_id="99999",
            name="Test Pitcher",
            year=2023,
            pa=50,  # Below min_pa
            xba=0.240,
            xslg=0.380,
            xwoba=0.310,
            xera=3.50,
            barrel_rate=0.06,
            hard_hit_rate=0.35,
        )

        features = extractor.extract(player, statcast, None)
        assert features is None

    def test_extract_reliever_has_zero_is_starter(self) -> None:
        extractor = PitcherFeatureExtractor(min_pa=50)

        player = PlayerRates(
            player_id="12345",
            name="Test Pitcher",
            year=2024,
            age=30,
            rates={
                "h": 0.080,
                "er": 0.030,
                "so": 0.090,
                "bb": 0.030,
                "hr": 0.010,
            },
            opportunities=200.0,
            metadata={"is_starter": False},
        )

        statcast = StatcastPitcherStats(
            player_id="99999",
            name="Test Pitcher",
            year=2023,
            pa=150,
            xba=0.240,
            xslg=0.380,
            xwoba=0.310,
            xera=3.50,
            barrel_rate=0.06,
            hard_hit_rate=0.35,
        )

        features = extractor.extract(player, statcast, None)

        assert features is not None
        assert features[20] == 0.0  # is_starter = False
