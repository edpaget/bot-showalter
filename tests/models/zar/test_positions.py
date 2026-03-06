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
        positions=positions or {"C": 1, "OF": 3},
    )


class TestBuildRosterSpots:
    def test_returns_league_positions(self) -> None:
        league = _league(positions={"C": 1, "OF": 3, "SS": 1})
        result = build_roster_spots(league)
        assert result == {"C": 1, "OF": 3, "SS": 1}

    def test_includes_util_when_roster_util_positive(self) -> None:
        league = _league(positions={"C": 1, "OF": 3}, roster_util=2)
        result = build_roster_spots(league)
        assert result == {"C": 1, "OF": 3, "UTIL": 2}

    def test_omits_util_when_roster_util_zero(self) -> None:
        league = _league(positions={"C": 1, "OF": 3}, roster_util=0)
        result = build_roster_spots(league)
        assert "UTIL" not in result

    def test_pitcher_override_returns_override_dict(self) -> None:
        league = _league(positions={"C": 1, "OF": 3})
        override = {"P": 5}
        result = build_roster_spots(league, pitcher_roster_spots=override)
        assert result == {"P": 5}

    def test_pitcher_override_ignores_league_positions(self) -> None:
        league = _league(positions={"C": 1, "OF": 3}, roster_util=2)
        override = {"SP": 3, "RP": 2}
        result = build_roster_spots(league, pitcher_roster_spots=override)
        assert result == {"SP": 3, "RP": 2}
        assert "C" not in result
        assert "UTIL" not in result


class TestBuildPositionMapMinGames:
    def test_default_min_games_keeps_all(self) -> None:
        """Default min_games=1 keeps all appearances with games >= 1."""
        league = _league(positions={"C": 1, "OF": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        result = build_position_map(appearances, league)
        assert 1 in result
        assert 2 in result

    def test_min_games_filters_low_appearances(self) -> None:
        """Appearances with games < min_games are excluded."""
        league = _league(positions={"C": 1, "OF": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=5),
            PositionAppearance(player_id=1, season=2025, position="OF", games=50),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        result = build_position_map(appearances, league, min_games=10)
        # Player 1's C appearances (5 games) excluded, OF (50) kept
        assert "C" not in result[1]
        assert "OF" in result[1]
        # Player 2 fully included
        assert "OF" in result[2]

    def test_min_games_removes_player_entirely(self) -> None:
        """Player with all appearances below min_games is excluded."""
        league = _league(positions={"C": 1, "OF": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=3),
            PositionAppearance(player_id=2, season=2025, position="OF", games=150),
        ]
        result = build_position_map(appearances, league, min_games=10)
        assert 1 not in result
        assert 2 in result

    def test_min_games_exact_threshold(self) -> None:
        """Appearances with games == min_games are included."""
        league = _league(positions={"C": 1, "OF": 3}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=10),
        ]
        result = build_position_map(appearances, league, min_games=10)
        assert 1 in result
        assert "C" in result[1]


class TestPositionMapKeysMatchLeagueKeys:
    def test_position_map_keys_match_league_position_keys(self) -> None:
        """Every position in build_position_map output is a league position or UTIL."""
        league = _league(positions={"C": 1, "OF": 3, "SS": 1}, roster_util=1)
        appearances = [
            PositionAppearance(player_id=1, season=2025, position="C", games=100),
            PositionAppearance(player_id=2, season=2025, position="LF", games=80),
            PositionAppearance(player_id=3, season=2025, position="SS", games=120),
            PositionAppearance(player_id=4, season=2025, position="RF", games=90),
        ]
        result = build_position_map(appearances, league)
        valid_keys = set(league.positions.keys()) | {"UTIL"}
        for positions in result.values():
            for pos in positions:
                assert pos in valid_keys, f"{pos!r} not in {valid_keys}"
