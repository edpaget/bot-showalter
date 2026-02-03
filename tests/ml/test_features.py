"""Tests for ML feature extraction."""

import numpy as np
import pytest

from fantasy_baseball_manager.ml.features import BatterFeatureExtractor, PitcherFeatureExtractor
from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
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
        assert len(names) == 18

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
        assert features.shape == (18,)
        assert features[0] == 0.035  # marcel_hr
        assert features[7] == 0.280  # xba
        assert features[12] == 28  # age
        assert features[17] == 500.0  # opportunities

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
        # barrel_vs_hr_ratio should be 0 when HR rate is 0
        assert features[16] == 0.0


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
