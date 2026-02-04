"""Unit tests for the agent tool functions."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from fantasy_baseball_manager.agent.tools import (
    compare_players,
    get_league_settings,
    get_player_info,
    lookup_player,
    project_batters,
    project_pitchers,
    rank_keepers,
)
from fantasy_baseball_manager.valuation.models import (
    CategoryValue,
    LeagueSettings,
    PlayerValue,
    ScoringStyle,
    StatCategory,
)


def _make_batter(player_id: str, name: str, hr: float, r: float, rbi: float) -> PlayerValue:
    """Create a mock batter PlayerValue."""
    total = hr + r + rbi
    return PlayerValue(
        player_id=player_id,
        name=name,
        category_values=(
            CategoryValue(category=StatCategory.HR, raw_stat=hr, value=hr / 10),
            CategoryValue(category=StatCategory.R, raw_stat=r, value=r / 10),
            CategoryValue(category=StatCategory.RBI, raw_stat=rbi, value=rbi / 10),
        ),
        total_value=total / 10,
        position_type="B",
    )


def _make_pitcher(player_id: str, name: str, k: float, era: float, whip: float) -> PlayerValue:
    """Create a mock pitcher PlayerValue."""
    # ERA/WHIP are negative contributors, so invert for z-score
    total = k / 10 - era - whip
    return PlayerValue(
        player_id=player_id,
        name=name,
        category_values=(
            CategoryValue(category=StatCategory.K, raw_stat=k, value=k / 10),
            CategoryValue(category=StatCategory.ERA, raw_stat=era, value=-era),
            CategoryValue(category=StatCategory.WHIP, raw_stat=whip, value=-whip),
        ),
        total_value=total,
        position_type="P",
    )


@pytest.fixture
def mock_projections() -> tuple[list[PlayerValue], dict[tuple[str, str], tuple[str, ...]]]:
    """Create mock projection data."""
    batters = [
        _make_batter("b1", "Aaron Judge", 45, 100, 110),
        _make_batter("b2", "Juan Soto", 35, 95, 90),
        _make_batter("b3", "Shohei Ohtani", 40, 90, 95),
    ]
    pitchers = [
        _make_pitcher("p1", "Corbin Burnes", 220, 2.90, 1.05),
        _make_pitcher("p2", "Spencer Strider", 250, 3.10, 1.10),
    ]
    all_values = batters + pitchers

    positions: dict[tuple[str, str], tuple[str, ...]] = {
        ("b1", "B"): ("OF",),
        ("b2", "B"): ("OF",),
        ("b3", "B"): ("DH", "OF"),
        ("p1", "P"): ("SP",),
        ("p2", "P"): ("SP",),
    }

    return all_values, positions


class TestProjectBatters:
    def test_returns_formatted_table(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"year": 2025, "engine": "marcel", "top_n": 10}
            result = project_batters.invoke(args)

        assert "Top" in result
        assert "Projected Batters" in result
        assert "Aaron Judge" in result
        assert "Juan Soto" in result

    def test_respects_top_n(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"year": 2025, "engine": "marcel", "top_n": 2}
            result = project_batters.invoke(args)

        # Should have 2 batters in output
        assert "Aaron Judge" in result
        # Ohtani might or might not be included depending on sort order
        lines = [line for line in result.split("\n") if line.strip() and not line.startswith("-")]
        # Subtract header lines - count actual player rows
        player_lines = [line for line in lines if "Judge" in line or "Soto" in line or "Ohtani" in line]
        assert len(player_lines) <= 2

    def test_invalid_engine_returns_error(self) -> None:
        args: dict[str, Any] = {"year": 2025, "engine": "invalid_engine", "top_n": 10}
        result = project_batters.invoke(args)
        assert "Invalid engine" in result
        assert "invalid_engine" in result


class TestProjectPitchers:
    def test_returns_formatted_table(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"year": 2025, "engine": "marcel", "top_n": 10}
            result = project_pitchers.invoke(args)

        assert "Top" in result
        assert "Projected Pitchers" in result
        assert "Corbin Burnes" in result


class TestLookupPlayer:
    def test_finds_player_by_name(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"name": "Judge"}
            result = lookup_player.invoke(args)

        assert "Aaron Judge" in result
        assert "Batter" in result
        assert "Total Z-Score Value" in result

    def test_partial_match_case_insensitive(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"name": "soto"}
            result = lookup_player.invoke(args)

        assert "Juan Soto" in result

    def test_not_found_returns_message(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"name": "Nonexistent Player"}
            result = lookup_player.invoke(args)

        assert "No players found" in result


class TestComparePlayers:
    def test_compares_multiple_players(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"names": "Judge, Soto"}
            result = compare_players.invoke(args)

        assert "Comparison" in result
        assert "Aaron Judge" in result or "Judge" in result
        assert "Juan Soto" in result or "Soto" in result

    def test_handles_not_found(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"names": "Judge, Unknown"}
            result = compare_players.invoke(args)

        assert "Could not find" in result
        assert "Unknown" in result

    def test_requires_two_names(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"names": "Judge"}
            result = compare_players.invoke(args)

        assert "at least two" in result


class TestRankKeepers:
    def test_ranks_candidates(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {
                "candidates": "Judge, Soto, Ohtani",
                "user_pick": 5,
                "teams": 12,
                "keeper_slots": 4,
            }
            result = rank_keepers.invoke(args)

        assert "Keeper Rankings" in result
        assert "Surplus" in result

    def test_handles_not_found(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {
                "candidates": "Judge, Unknown",
                "user_pick": 5,
                "teams": 12,
                "keeper_slots": 4,
            }
            result = rank_keepers.invoke(args)

        assert "Could not find" in result or "Aaron Judge" in result


class TestGetLeagueSettings:
    def test_returns_settings(self) -> None:
        mock_settings = LeagueSettings(
            team_count=12,
            batting_categories=(StatCategory.HR, StatCategory.R, StatCategory.RBI),
            pitching_categories=(StatCategory.K, StatCategory.ERA, StatCategory.WHIP),
            scoring_style=ScoringStyle.H2H_EACH_CATEGORY,
        )

        with patch(
            "fantasy_baseball_manager.agent.tools.load_league_settings",
            return_value=mock_settings,
        ):
            args: dict[str, Any] = {}
            result = get_league_settings.invoke(args)

        assert "League Settings" in result
        assert "Teams: 12" in result
        assert "HR" in result
        assert "ERA" in result


class TestGetPlayerInfo:
    def test_returns_player_info(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        mock_mlb_response = {
            "fullName": "Aaron Judge",
            "currentTeam": {"name": "New York Yankees"},
            "primaryPosition": {"name": "Outfielder", "abbreviation": "RF"},
            "currentAge": 32,
            "batSide": {"code": "R"},
            "pitchHand": {"code": "R"},
            "height": "6' 7\"",
            "weight": 282,
            "birthCity": "Linden",
            "birthStateProvince": "CA",
            "birthCountry": "USA",
            "mlbDebutDate": "2016-08-13",
        }

        mock_mapper = type("MockMapper", (), {"fangraphs_to_mlbam": lambda _self, _fid: "592450"})()

        with (
            patch(
                "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
                return_value=(all_values, positions),
            ),
            patch(
                "fantasy_baseball_manager.services.container.get_container",
                return_value=type("MockContainer", (), {"id_mapper": mock_mapper})(),
            ),
            patch(
                "fantasy_baseball_manager.agent.tools._fetch_mlb_player_info",
                return_value=mock_mlb_response,
            ),
        ):
            args: dict[str, Any] = {"name": "Judge"}
            result = get_player_info.invoke(args)

        assert "Aaron Judge" in result
        assert "New York Yankees" in result
        assert "Outfielder" in result
        assert "Age: 32" in result
        assert "Bats/Throws: R/R" in result

    def test_player_not_in_projections(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        with patch(
            "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
            return_value=(all_values, positions),
        ):
            args: dict[str, Any] = {"name": "Nonexistent Player"}
            result = get_player_info.invoke(args)

        assert "No players found" in result

    def test_no_mlbam_id(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        mock_mapper = type("MockMapper", (), {"fangraphs_to_mlbam": lambda _self, _fid: None})()

        with (
            patch(
                "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
                return_value=(all_values, positions),
            ),
            patch(
                "fantasy_baseball_manager.services.container.get_container",
                return_value=type("MockContainer", (), {"id_mapper": mock_mapper})(),
            ),
        ):
            args: dict[str, Any] = {"name": "Judge"}
            result = get_player_info.invoke(args)

        assert "could not find their MLB ID" in result

    def test_mlb_api_failure(self, mock_projections: tuple) -> None:
        all_values, positions = mock_projections

        mock_mapper = type("MockMapper", (), {"fangraphs_to_mlbam": lambda _self, _fid: "592450"})()

        with (
            patch(
                "fantasy_baseball_manager.agent.tools.build_projections_and_positions",
                return_value=(all_values, positions),
            ),
            patch(
                "fantasy_baseball_manager.services.container.get_container",
                return_value=type("MockContainer", (), {"id_mapper": mock_mapper})(),
            ),
            patch(
                "fantasy_baseball_manager.agent.tools._fetch_mlb_player_info",
                return_value=None,
            ),
        ):
            args: dict[str, Any] = {"name": "Judge"}
            result = get_player_info.invoke(args)

        assert "Could not retrieve current info" in result
