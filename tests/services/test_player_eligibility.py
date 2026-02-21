from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.services.player_eligibility import PlayerEligibilityService
from tests.fakes.repos import FakePositionAppearanceRepo


def _league(
    positions: dict[str, int] | None = None,
    roster_util: int = 1,
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

    def test_does_not_fall_back_when_current_season_has_data(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=100),
            PositionAppearance(player_id=2, season=2024, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        # Only player 1 from 2025 should appear, not player 2 from 2024
        assert 1 in result
        assert 2 not in result


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

    def test_default_min_games_filters_below_10(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        # Player 1 has only 5 games at C, below default min_games=10
        assert 1 not in result
        assert 2 in result

    def test_custom_min_games(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league(), min_games=3)
        # Player 1 has 5 games >= 3, should be included
        assert 1 in result
        assert 2 in result

    def test_min_games_applied_to_fallback_data(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2026, _league())
        # Fallback to 2025; player 1 still filtered out by min_games=10
        assert 1 not in result
        assert 2 in result

    def test_multi_position_player_partial_filter(self) -> None:
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=1, season=2025, position="OF", games=50),
        ]
        service = PlayerEligibilityService(FakePositionAppearanceRepo(appearances))
        result = service.get_batter_positions(2025, _league())
        # C filtered out (5 < 10), OF kept (50 >= 10)
        assert 1 in result
        assert "c" not in result[1]
        assert "of" in result[1]
