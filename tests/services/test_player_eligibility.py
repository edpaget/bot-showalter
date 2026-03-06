from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    EligibilityRules,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from tests.fakes.repos import FakePitchingStatsRepo, FakePositionAppearanceRepo


def _league(
    positions: dict[str, int] | None = None,
    roster_util: int = 1,
    pitcher_positions: dict[str, int] | None = None,
    eligibility: EligibilityRules | None = None,
) -> LeagueSettings:
    return LeagueSettings(
        name="Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=3,
        roster_pitchers=2,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="W", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=roster_util,
        positions=positions or {"c": 1, "of": 3},
        pitcher_positions=pitcher_positions or {},
        eligibility=eligibility or EligibilityRules(),
    )


class TestGetBatterPositionsCurrentSeason:
    """When position data exists for the target season, it is used directly."""

    def test_returns_positions_from_target_season(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=100),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        assert 1 in result
        assert "c" in result[1]
        assert 2 in result
        assert "of" in result[2]

    def test_prior_season_data_also_included(self) -> None:
        """Multi-season: player 2 from 2024 SHOULD appear when querying 2025."""
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=100),
            PositionAppearance(player_id=2, season=2024, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        assert 1 in result
        assert 2 in result
        assert "of" in result[2]


class TestGetBatterPositionsFallback:
    """When position data is missing for the target season, fall back to season - 1."""

    def test_falls_back_to_previous_season(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=100),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2026, _league())
        assert 1 in result
        assert "c" in result[1]
        assert 2 in result
        assert "of" in result[2]

    def test_returns_empty_when_no_data_in_either_season(self) -> None:
        service = PlayerEligibilityService(FakePositionAppearanceRepo([]))
        result = service.get_batter_positions(2026, _league())
        assert result == {}


class TestGetBatterPositionsMinGames:
    """Positions with fewer than min_games are excluded."""

    def test_default_min_games_filters_below_5(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=3),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        # Player 1 has only 3 games at C, below default min_games=5
        assert 1 not in result
        assert 2 in result

    def test_custom_min_games(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        league = _league(eligibility=EligibilityRules(batter_min_games=3))
        result = service.get_batter_positions(2025, league)
        # Player 1 has 5 games >= 3, should be included
        assert 1 in result
        assert 2 in result

    def test_min_games_applied_to_fallback_data(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=3),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2026, _league())
        # Fallback to 2025; player 1 still filtered out by min_games=5
        assert 1 not in result
        assert 2 in result

    def test_multi_position_player_partial_filter(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=3),
            PositionAppearance(player_id=1, season=2025, position="OF", games=50),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        # C filtered out (3 < 5), OF kept (50 >= 5)
        assert 1 in result
        assert "c" not in result[1]
        assert "of" in result[1]


class TestGetBatterPositionsCarryover:
    """Prior-season eligibility carries forward (Yahoo-style)."""

    def test_prior_season_only_carries_forward(self) -> None:
        """Player has OF in 2024 (>= 5 games), no data in 2025 → eligible at OF for 2025."""
        appearances = [
            PositionAppearance(player_id=1, season=2024, position="OF", games=65),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        assert 1 in result
        assert "of" in result[1]

    def test_union_of_both_seasons(self) -> None:
        """Player has C in 2024, OF in 2025 → eligible at both for 2025."""
        appearances = [
            PositionAppearance(player_id=1, season=2024, position="C", games=50),
            PositionAppearance(player_id=1, season=2025, position="OF", games=30),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        assert 1 in result
        assert "c" in result[1]
        assert "of" in result[1]

    def test_preseason_uses_prior_season(self) -> None:
        """No data for 2026 or 2025, data in 2025 used as sole source for 2026."""
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=100),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2026, _league())
        assert 1 in result
        assert "c" in result[1]


# ---------------------------------------------------------------------------
# Pitcher positions
# ---------------------------------------------------------------------------

_SP_RP_LEAGUE = {"sp": 2, "rp": 2, "p": 4}


def _pitching_stats(
    player_id: int,
    season: int,
    g: int,
    gs: int,
    source: str = "fangraphs",
) -> PitchingStats:
    return PitchingStats(player_id=player_id, season=season, source=source, g=g, gs=gs)


class TestGetPitcherPositionsBackwardCompat:
    """Empty pitcher_positions → all pitchers get ["p"]."""

    def test_empty_pitcher_positions_returns_p_for_all(self) -> None:
        service = PlayerEligibilityService(FakePositionAppearanceRepo())
        league = _league()
        result = service.get_pitcher_positions(2025, league, [10, 20])
        assert result == {10: ["p"], 20: ["p"]}


class TestGetPitcherPositionsSPOnly:
    """Pitcher with only starts → SP-eligible + P flex."""

    def test_sp_only(self) -> None:
        stats = [_pitching_stats(10, 2025, g=30, gs=30)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]
        assert "rp" not in result[10]
        assert "p" in result[10]


class TestGetPitcherPositionsRPOnly:
    """Pitcher with only relief appearances → RP-eligible + P flex."""

    def test_rp_only(self) -> None:
        stats = [_pitching_stats(10, 2025, g=60, gs=0)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "rp" in result[10]
        assert "sp" not in result[10]
        assert "p" in result[10]


class TestGetPitcherPositionsDualEligible:
    """Pitcher with both starts and relief → SP + RP + P."""

    def test_dual_eligible(self) -> None:
        stats = [_pitching_stats(10, 2025, g=40, gs=20)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]
        assert "rp" in result[10]
        assert "p" in result[10]


class TestGetPitcherPositionsSeasonFallback:
    """When no pitching stats for current season, falls back to season - 1."""

    def test_falls_back_to_previous_season(self) -> None:
        stats = [_pitching_stats(10, 2024, g=30, gs=30)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]
        assert "p" in result[10]


class TestGetPitcherPositionsRookieWithoutStats:
    """Pitcher in pitcher_ids but with no stats → gets ["p"] if "p" in config."""

    def test_rookie_gets_flex(self) -> None:
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [99])
        assert result[99] == ["p"]

    def test_rookie_no_flex_slot(self) -> None:
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(),
        )
        league = _league(pitcher_positions={"sp": 2, "rp": 2})
        result = service.get_pitcher_positions(2025, league, [99])
        assert result[99] == []


class TestGetPitcherPositionsMultiSource:
    """Stats from multiple sources → aggregate max(g), max(gs)."""

    def test_aggregates_across_sources(self) -> None:
        stats = [
            _pitching_stats(10, 2025, g=20, gs=15, source="fangraphs"),
            _pitching_stats(10, 2025, g=25, gs=10, source="baseball-reference"),
        ]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        # max(g)=25, max(gs)=15 → gs>=3 so SP, (25-15)=10>=5 so RP
        assert "sp" in result[10]
        assert "rp" in result[10]
        assert "p" in result[10]


class TestGetPitcherPositionsFilteredByConfig:
    """Only positions present in league.pitcher_positions are included."""

    def test_sp_only_league(self) -> None:
        stats = [_pitching_stats(10, 2025, g=40, gs=20)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions={"sp": 4})
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]
        assert "rp" not in result[10]
        assert "p" not in result[10]


class TestGetPitcherPositionsYahooThresholds:
    """Yahoo thresholds: SP requires gs>=3, RP requires (g-gs)>=5."""

    def test_below_sp_threshold(self) -> None:
        """gs=2 → NOT SP-eligible."""
        stats = [_pitching_stats(10, 2025, g=60, gs=2)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" not in result[10]
        assert "rp" in result[10]

    def test_below_rp_threshold(self) -> None:
        """g=30, gs=27 (3 relief apps) → NOT RP-eligible."""
        stats = [_pitching_stats(10, 2025, g=30, gs=27)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]
        assert "rp" not in result[10]

    def test_at_sp_threshold(self) -> None:
        """gs=3 → exactly at threshold, SP-eligible."""
        stats = [_pitching_stats(10, 2025, g=3, gs=3)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]

    def test_at_rp_threshold(self) -> None:
        """g=5, gs=0 → exactly 5 relief apps, RP-eligible."""
        stats = [_pitching_stats(10, 2025, g=5, gs=0)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "rp" in result[10]


class TestGetPitcherPositionsCarryover:
    """Pitcher multi-season carryover."""

    def test_prior_season_only(self) -> None:
        """Stats only in 2024 → used for 2025 classification."""
        stats = [_pitching_stats(10, 2024, g=30, gs=25)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]
        assert "rp" in result[10]

    def test_multi_season_takes_max(self) -> None:
        """Stats in both years, takes max g and max gs."""
        stats = [
            _pitching_stats(10, 2024, g=20, gs=18),
            _pitching_stats(10, 2025, g=10, gs=2),
        ]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(pitcher_positions=_SP_RP_LEAGUE)
        result = service.get_pitcher_positions(2025, league, [10])
        # max(g)=20, max(gs)=18 → gs>=3 so SP, (20-18)=2 < 5 so NOT RP
        assert "sp" in result[10]
        assert "rp" not in result[10]


# ---------------------------------------------------------------------------
# Configurable eligibility rules
# ---------------------------------------------------------------------------


class TestCarryoverSeasons:
    """carryover_seasons controls how many prior seasons are included."""

    def test_carryover_zero_disables_multi_season(self) -> None:
        """With carryover_seasons=0, only current season data is used."""
        appearances = [
            PositionAppearance(player_id=1, season=2024, position="OF", games=65),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        league = _league(eligibility=EligibilityRules(carryover_seasons=0))
        result = service.get_batter_positions(2025, league)
        # No data for 2025, and carryover disabled → empty
        assert 1 not in result

    def test_carryover_two_looks_back_two_years(self) -> None:
        """With carryover_seasons=2, data from season-2 is included."""
        appearances = [
            PositionAppearance(player_id=1, season=2023, position="C", games=100),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        league = _league(eligibility=EligibilityRules(carryover_seasons=2))
        result = service.get_batter_positions(2025, league)
        assert 1 in result
        assert "c" in result[1]

    def test_carryover_one_does_not_reach_two_years_back(self) -> None:
        """Default carryover_seasons=1 does NOT include season-2."""
        appearances = [
            PositionAppearance(player_id=1, season=2023, position="C", games=100),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        league = _league(eligibility=EligibilityRules(carryover_seasons=1))
        result = service.get_batter_positions(2025, league)
        assert 1 not in result

    def test_pitcher_carryover_zero(self) -> None:
        """Pitcher with carryover_seasons=0 ignores prior season stats."""
        stats = [_pitching_stats(10, 2024, g=30, gs=30)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(
            pitcher_positions=_SP_RP_LEAGUE,
            eligibility=EligibilityRules(carryover_seasons=0),
        )
        result = service.get_pitcher_positions(2025, league, [10])
        # No data for 2025, carryover disabled → rookie fallback
        assert result[10] == ["p"]

    def test_pitcher_carryover_two(self) -> None:
        """Pitcher with carryover_seasons=2 includes season-2 stats."""
        stats = [_pitching_stats(10, 2023, g=30, gs=30)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(
            pitcher_positions=_SP_RP_LEAGUE,
            eligibility=EligibilityRules(carryover_seasons=2),
        )
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]


class TestCustomPitcherThresholds:
    """Custom sp_min_starts and rp_min_relief override defaults."""

    def test_custom_sp_min_starts(self) -> None:
        """With sp_min_starts=5, pitcher with gs=4 is NOT SP-eligible."""
        stats = [_pitching_stats(10, 2025, g=30, gs=4)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(
            pitcher_positions=_SP_RP_LEAGUE,
            eligibility=EligibilityRules(sp_min_starts=5),
        )
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" not in result[10]
        assert "rp" in result[10]

    def test_custom_rp_min_relief(self) -> None:
        """With rp_min_relief=10, pitcher with 8 relief apps is NOT RP-eligible."""
        stats = [_pitching_stats(10, 2025, g=20, gs=12)]
        service = PlayerEligibilityService(
            FakePositionAppearanceRepo(),
            pitching_stats_repo=FakePitchingStatsRepo(stats),
        )
        league = _league(
            pitcher_positions=_SP_RP_LEAGUE,
            eligibility=EligibilityRules(rp_min_relief=10),
        )
        result = service.get_pitcher_positions(2025, league, [10])
        assert "sp" in result[10]
        assert "rp" not in result[10]
