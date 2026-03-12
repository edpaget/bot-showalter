"""Tests for pitcher rate-stat bias correction."""

import pytest

from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.models.ensemble.rate_calibration import (
    RateCalibrationConfig,
    StatCalibration,
    calibrate_pitcher_rates,
    compute_prior_season_counts,
)
from tests.fakes.repos import FakePitchingStatsRepo


class TestRateCalibrationConfigFromParams:
    def test_parses_era_and_whip(self) -> None:
        params = {
            "era_intercept": 0.6805,
            "era_slope": 0.789,
            "era_prior_coef": 0.0252,
            "whip_intercept": 0.2674,
            "whip_slope": 0.7543,
            "whip_prior_coef": 0.005,
        }
        config = RateCalibrationConfig.from_params(params)
        assert "era" in config.stats
        assert "whip" in config.stats
        assert config.stats["era"].intercept == pytest.approx(0.6805)
        assert config.stats["era"].slope == pytest.approx(0.789)
        assert config.stats["era"].prior_coef == pytest.approx(0.0252)

    def test_custom_min_ip_and_lag_seasons(self) -> None:
        params = {
            "min_ip": 20.0,
            "lag_seasons": [1, 2, 3],
            "era_intercept": 0.5,
            "era_slope": 0.8,
            "era_prior_coef": 0.01,
        }
        config = RateCalibrationConfig.from_params(params)
        assert config.min_ip == 20.0
        assert config.lag_seasons == (1, 2, 3)

    def test_incomplete_stat_ignored(self) -> None:
        """A stat with only some coefficients is not included."""
        params = {
            "era_intercept": 0.5,
            "era_slope": 0.8,
            # missing era_prior_coef
        }
        config = RateCalibrationConfig.from_params(params)
        assert "era" not in config.stats

    def test_empty_params(self) -> None:
        config = RateCalibrationConfig.from_params({})
        assert config.stats == {}
        assert config.min_ip == 10.0
        assert config.lag_seasons == (1, 2)


class TestComputePriorSeasonCounts:
    def test_counts_qualifying_seasons(self) -> None:
        stats = [
            PitchingStats(player_id=1, season=2023, source="fg", ip=150.0),
            PitchingStats(player_id=1, season=2022, source="fg", ip=100.0),
            PitchingStats(player_id=2, season=2023, source="fg", ip=50.0),
            # player 2 has no 2022 stats
        ]
        repo = FakePitchingStatsRepo(stats)
        result = compute_prior_season_counts({1, 2}, season=2024, pitching_stats_repo=repo)
        assert result[1] == 2  # qualified in both 2023 and 2022
        assert result[2] == 1  # only 2023

    def test_below_min_ip_not_counted(self) -> None:
        stats = [
            PitchingStats(player_id=1, season=2023, source="fg", ip=5.0),  # below threshold
        ]
        repo = FakePitchingStatsRepo(stats)
        result = compute_prior_season_counts({1}, season=2024, pitching_stats_repo=repo)
        assert result[1] == 0

    def test_no_prior_data(self) -> None:
        repo = FakePitchingStatsRepo([])
        result = compute_prior_season_counts({1, 2}, season=2024, pitching_stats_repo=repo)
        assert result[1] == 0
        assert result[2] == 0

    def test_custom_min_ip(self) -> None:
        stats = [
            PitchingStats(player_id=1, season=2023, source="fg", ip=15.0),
        ]
        repo = FakePitchingStatsRepo(stats)
        result = compute_prior_season_counts({1}, season=2024, pitching_stats_repo=repo, min_ip=20.0)
        assert result[1] == 0

    def test_custom_lag_seasons(self) -> None:
        stats = [
            PitchingStats(player_id=1, season=2021, source="fg", ip=100.0),
        ]
        repo = FakePitchingStatsRepo(stats)
        # With lag_seasons=(1, 2), 2021 is not covered for season 2024
        result = compute_prior_season_counts({1}, season=2024, pitching_stats_repo=repo, lag_seasons=(1, 2))
        assert result[1] == 0
        # With lag_seasons=(1, 2, 3), 2021 is covered
        result = compute_prior_season_counts({1}, season=2024, pitching_stats_repo=repo, lag_seasons=(1, 2, 3))
        assert result[1] == 1

    def test_null_ip_not_counted(self) -> None:
        stats = [
            PitchingStats(player_id=1, season=2023, source="fg", ip=None),
        ]
        repo = FakePitchingStatsRepo(stats)
        result = compute_prior_season_counts({1}, season=2024, pitching_stats_repo=repo)
        assert result[1] == 0


class TestCalibratePitcherRates:
    def test_corrects_pitcher_rate_stats(self) -> None:
        config = RateCalibrationConfig(
            stats={
                "era": StatCalibration(intercept=0.5, slope=0.8, prior_coef=0.02),
            }
        )
        predictions = [
            {"player_id": 1, "player_type": "pitcher", "era": 4.0, "so": 200},
        ]
        prior_counts = {1: 2}
        calibrate_pitcher_rates(predictions, prior_counts, config)
        # corrected = 0.5 + 0.8 * 4.0 + 0.02 * 2 = 0.5 + 3.2 + 0.04 = 3.74
        assert predictions[0]["era"] == pytest.approx(3.74)
        # counting stat untouched
        assert predictions[0]["so"] == 200

    def test_skips_batters(self) -> None:
        config = RateCalibrationConfig(stats={"era": StatCalibration(intercept=0.5, slope=0.8, prior_coef=0.02)})
        predictions = [
            {"player_id": 1, "player_type": "batter", "era": 4.0},
        ]
        calibrate_pitcher_rates(predictions, {1: 1}, config)
        assert predictions[0]["era"] == 4.0  # unchanged

    def test_missing_stat_untouched(self) -> None:
        config = RateCalibrationConfig(stats={"era": StatCalibration(intercept=0.5, slope=0.8, prior_coef=0.02)})
        predictions = [
            {"player_id": 1, "player_type": "pitcher", "whip": 1.2},
        ]
        calibrate_pitcher_rates(predictions, {1: 1}, config)
        assert predictions[0]["whip"] == 1.2  # unchanged, "era" not present

    def test_zero_prior_uses_zero(self) -> None:
        config = RateCalibrationConfig(stats={"era": StatCalibration(intercept=0.5, slope=0.8, prior_coef=0.02)})
        predictions = [
            {"player_id": 1, "player_type": "pitcher", "era": 4.0},
        ]
        calibrate_pitcher_rates(predictions, {}, config)
        # n_prior defaults to 0: 0.5 + 0.8 * 4.0 + 0.02 * 0 = 3.7
        assert predictions[0]["era"] == pytest.approx(3.7)

    def test_multiple_stats_corrected(self) -> None:
        config = RateCalibrationConfig(
            stats={
                "era": StatCalibration(intercept=0.5, slope=0.8, prior_coef=0.02),
                "whip": StatCalibration(intercept=0.2, slope=0.75, prior_coef=0.005),
            }
        )
        predictions = [
            {"player_id": 1, "player_type": "pitcher", "era": 4.0, "whip": 1.3},
        ]
        calibrate_pitcher_rates(predictions, {1: 1}, config)
        assert predictions[0]["era"] == pytest.approx(0.5 + 0.8 * 4.0 + 0.02 * 1)
        assert predictions[0]["whip"] == pytest.approx(0.2 + 0.75 * 1.3 + 0.005 * 1)
