"""Tests for compute_historical_stdevs."""

import math
import statistics

import pytest

from fantasy_baseball_manager.domain import BattingStats, PitchingStats
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.services.historical_stdev import compute_historical_stdevs
from tests.fakes.repos import FakeBattingStatsRepo, FakePitchingStatsRepo


def _league(
    batting_categories: tuple[CategoryConfig, ...] = (),
    pitching_categories: tuple[CategoryConfig, ...] = (),
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.ROTO,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=9,
        batting_categories=batting_categories,
        pitching_categories=pitching_categories,
    )


HR_CAT = CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
SB_CAT = CategoryConfig(key="sb", name="SB", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
AVG_CAT = CategoryConfig(
    key="avg",
    name="AVG",
    stat_type=StatType.RATE,
    direction=Direction.HIGHER,
    numerator="h",
    denominator="ab",
)
ERA_CAT = CategoryConfig(
    key="era",
    name="ERA",
    stat_type=StatType.RATE,
    direction=Direction.LOWER,
    numerator="er",
    denominator="ip",
)
SO_PITCH_CAT = CategoryConfig(key="so", name="SO", stat_type=StatType.COUNTING, direction=Direction.HIGHER)


class TestCountingStatStdev:
    """Test 1: Counting stat stdev (single season, hand-calculated)."""

    def test_counting_stat_stdev_single_season(self) -> None:
        batters = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=500, hr=30, ab=450),
            BattingStats(player_id=2, season=2024, source="fangraphs", pa=500, hr=20, ab=450),
            BattingStats(player_id=3, season=2024, source="fangraphs", pa=500, hr=10, ab=450),
        ]
        league = _league(batting_categories=(HR_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters),
            pitching_repo=FakePitchingStatsRepo(),
        )
        expected = statistics.pstdev([30.0, 20.0, 10.0])
        assert result["hr"] == pytest.approx(expected)


class TestRateStatStdev:
    """Test 2: Rate stat stdev differs from raw stdev (validates marginal-contribution conversion)."""

    def test_rate_stat_stdev_differs_from_raw(self) -> None:
        # Two batters with different AB but same AVG will have different marginal contributions
        batters = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=600, ab=500, h=150),  # .300 AVG
            BattingStats(player_id=2, season=2024, source="fangraphs", pa=400, ab=300, h=90),  # .300 AVG
            BattingStats(player_id=3, season=2024, source="fangraphs", pa=500, ab=400, h=100),  # .250 AVG
        ]
        league = _league(batting_categories=(AVG_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters),
            pitching_repo=FakePitchingStatsRepo(),
        )
        # Raw AVG stdev
        raw_stdev = statistics.pstdev([0.300, 0.300, 0.250])
        # Converted stdev should differ because marginal contributions weight by AB
        assert result["avg"] != pytest.approx(raw_stdev, abs=1e-6)
        assert result["avg"] > 0


class TestMultiSeasonAveraging:
    """Test 3: Multi-season averaging."""

    def test_averages_stdevs_across_seasons(self) -> None:
        batters_2023 = [
            BattingStats(player_id=1, season=2023, source="fangraphs", pa=500, hr=40, ab=450),
            BattingStats(player_id=2, season=2023, source="fangraphs", pa=500, hr=10, ab=450),
        ]
        batters_2024 = [
            BattingStats(player_id=3, season=2024, source="fangraphs", pa=500, hr=30, ab=450),
            BattingStats(player_id=4, season=2024, source="fangraphs", pa=500, hr=20, ab=450),
        ]
        league = _league(batting_categories=(HR_CAT,))
        result = compute_historical_stdevs(
            seasons=[2023, 2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters_2023 + batters_2024),
            pitching_repo=FakePitchingStatsRepo(),
        )
        stdev_2023 = statistics.pstdev([40.0, 10.0])
        stdev_2024 = statistics.pstdev([30.0, 20.0])
        expected = (stdev_2023 + stdev_2024) / 2
        assert result["hr"] == pytest.approx(expected)


class TestSingleSeason:
    """Test 4: Single season (no averaging edge case)."""

    def test_single_season_returns_that_season_stdev(self) -> None:
        batters = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=500, hr=25, ab=450),
            BattingStats(player_id=2, season=2024, source="fangraphs", pa=500, hr=15, ab=450),
        ]
        league = _league(batting_categories=(HR_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters),
            pitching_repo=FakePitchingStatsRepo(),
        )
        expected = statistics.pstdev([25.0, 15.0])
        assert result["hr"] == pytest.approx(expected)


class TestEmptySeasonExcluded:
    """Test 5: Empty season excluded from average."""

    def test_empty_season_excluded(self) -> None:
        batters_2023 = [
            BattingStats(player_id=1, season=2023, source="fangraphs", pa=500, hr=30, ab=450),
            BattingStats(player_id=2, season=2023, source="fangraphs", pa=500, hr=10, ab=450),
        ]
        # No data for 2024
        league = _league(batting_categories=(HR_CAT,))
        result = compute_historical_stdevs(
            seasons=[2023, 2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters_2023),
            pitching_repo=FakePitchingStatsRepo(),
        )
        # Should only use 2023 stdev, not average with 0
        expected = statistics.pstdev([30.0, 10.0])
        assert result["hr"] == pytest.approx(expected)


class TestZeroVariance:
    """Test 6: Zero-variance category returns 0.0."""

    def test_zero_variance_returns_zero(self) -> None:
        batters = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=500, hr=20, ab=450),
            BattingStats(player_id=2, season=2024, source="fangraphs", pa=500, hr=20, ab=450),
        ]
        league = _league(batting_categories=(HR_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters),
            pitching_repo=FakePitchingStatsRepo(),
        )
        assert result["hr"] == 0.0


class TestCombinedBattingPitching:
    """Test 7: Combined batting + pitching categories in one call."""

    def test_combined_categories(self) -> None:
        batters = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=500, hr=30, ab=450, h=135),
            BattingStats(player_id=2, season=2024, source="fangraphs", pa=500, hr=10, ab=450, h=120),
        ]
        pitchers = [
            PitchingStats(player_id=3, season=2024, source="fangraphs", ip=180.0, er=60, so=200),
            PitchingStats(player_id=4, season=2024, source="fangraphs", ip=160.0, er=70, so=150),
        ]
        league = _league(
            batting_categories=(HR_CAT,),
            pitching_categories=(SO_PITCH_CAT,),
        )
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters),
            pitching_repo=FakePitchingStatsRepo(pitchers),
        )
        assert "hr" in result
        assert "so" in result
        assert result["hr"] == pytest.approx(statistics.pstdev([30.0, 10.0]))
        assert result["so"] == pytest.approx(statistics.pstdev([200.0, 150.0]))


class TestZeroPAAndIPExcluded:
    """Test 8: Zero-PA batters and zero-IP pitchers excluded."""

    def test_zero_pa_batters_excluded(self) -> None:
        batters = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=500, hr=30, ab=450),
            BattingStats(player_id=2, season=2024, source="fangraphs", pa=0, hr=0, ab=0),
            BattingStats(player_id=3, season=2024, source="fangraphs", pa=500, hr=10, ab=450),
        ]
        league = _league(batting_categories=(HR_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters),
            pitching_repo=FakePitchingStatsRepo(),
        )
        # Player 2 should be excluded (pa=0), stdev computed from players 1 and 3 only
        expected = statistics.pstdev([30.0, 10.0])
        assert result["hr"] == pytest.approx(expected)

    def test_zero_ip_pitchers_excluded(self) -> None:
        pitchers = [
            PitchingStats(player_id=1, season=2024, source="fangraphs", ip=180.0, so=200),
            PitchingStats(player_id=2, season=2024, source="fangraphs", ip=0.0, so=0),
            PitchingStats(player_id=3, season=2024, source="fangraphs", ip=160.0, so=150),
        ]
        league = _league(pitching_categories=(SO_PITCH_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(),
            pitching_repo=FakePitchingStatsRepo(pitchers),
        )
        expected = statistics.pstdev([200.0, 150.0])
        assert result["so"] == pytest.approx(expected)

    def test_none_pa_batters_excluded(self) -> None:
        batters = [
            BattingStats(player_id=1, season=2024, source="fangraphs", pa=500, hr=30, ab=450),
            BattingStats(player_id=2, season=2024, source="fangraphs", hr=5, ab=10),  # pa=None
            BattingStats(player_id=3, season=2024, source="fangraphs", pa=500, hr=10, ab=450),
        ]
        league = _league(batting_categories=(HR_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(batters),
            pitching_repo=FakePitchingStatsRepo(),
        )
        expected = statistics.pstdev([30.0, 10.0])
        assert result["hr"] == pytest.approx(expected)


class TestEraStdevUsesMarginalContribution:
    """Verify ERA stdev uses marginal-contribution conversion, not raw ERA values."""

    def test_era_stdev_uses_converted_values(self) -> None:
        pitchers = [
            PitchingStats(player_id=1, season=2024, source="fangraphs", ip=200.0, er=60),  # 2.70 ERA
            PitchingStats(player_id=2, season=2024, source="fangraphs", ip=150.0, er=50),  # 3.00 ERA
            PitchingStats(player_id=3, season=2024, source="fangraphs", ip=100.0, er=40),  # 3.60 ERA
        ]
        league = _league(pitching_categories=(ERA_CAT,))
        result = compute_historical_stdevs(
            seasons=[2024],
            league=league,
            batting_repo=FakeBattingStatsRepo(),
            pitching_repo=FakePitchingStatsRepo(pitchers),
        )
        # Should NOT equal raw ERA pstdev
        raw_eras = [60 * 9 / 200, 50 * 9 / 150, 40 * 9 / 100]
        raw_stdev = statistics.pstdev(raw_eras)
        assert result["era"] != pytest.approx(raw_stdev, abs=1e-6)
        assert result["era"] > 0
        assert math.isfinite(result["era"])
