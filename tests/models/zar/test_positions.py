from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.models.zar.positions import build_position_map, build_roster_spots


def _league(
    positions: dict[str, int] | None = None,
    roster_util: int = 0,
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


class TestBuildRosterSpots:
    def test_returns_league_positions(self) -> None:
        league = _league(positions={"c": 1, "of": 3, "ss": 1})
        result = build_roster_spots(league)
        assert result == {"c": 1, "of": 3, "ss": 1}

    def test_includes_util_when_roster_util_positive(self) -> None:
        league = _league(positions={"c": 1, "of": 3}, roster_util=2)
        result = build_roster_spots(league)
        assert result == {"c": 1, "of": 3, "util": 2}

    def test_omits_util_when_roster_util_zero(self) -> None:
        league = _league(positions={"c": 1, "of": 3}, roster_util=0)
        result = build_roster_spots(league)
        assert "util" not in result

    def test_pitcher_override_returns_override_dict(self) -> None:
        league = _league(positions={"c": 1, "of": 3})
        override = {"p": 5}
        result = build_roster_spots(league, pitcher_roster_spots=override)
        assert result == {"p": 5}

    def test_pitcher_override_ignores_league_positions(self) -> None:
        league = _league(positions={"c": 1, "of": 3}, roster_util=2)
        override = {"sp": 3, "rp": 2}
        result = build_roster_spots(league, pitcher_roster_spots=override)
        assert result == {"sp": 3, "rp": 2}
        assert "c" not in result
        assert "util" not in result


class TestBuildPositionMapMinGames:
    def test_default_min_games_keeps_all(self) -> None:
        """Default min_games=1 keeps all appearances with games >= 1."""
        league = _league(positions={"c": 1, "of": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        result = build_position_map(appearances, league)
        assert 1 in result
        assert 2 in result

    def test_min_games_filters_low_appearances(self) -> None:
        """Appearances with games < min_games are excluded."""
        league = _league(positions={"c": 1, "of": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=1, season=2025, position="OF", games=50),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        result = build_position_map(appearances, league, min_games=10)
        # Player 1's C appearances (5 games) excluded, OF (50) kept
        assert "c" not in result[1]
        assert "of" in result[1]
        # Player 2 fully included
        assert "of" in result[2]

    def test_min_games_removes_player_entirely(self) -> None:
        """Player with all appearances below min_games is excluded."""
        league = _league(positions={"c": 1, "of": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=3),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        result = build_position_map(appearances, league, min_games=10)
        assert 1 not in result
        assert 2 in result

    def test_min_games_exact_threshold(self) -> None:
        """Appearances with games == min_games are included."""
        league = _league(positions={"c": 1, "of": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=10),
        ]
        result = build_position_map(appearances, league, min_games=10)
        assert 1 in result
        assert "c" in result[1]
