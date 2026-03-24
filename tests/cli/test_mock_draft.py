"""Tests for mock draft CLI commands."""

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.domain import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def _mock_league() -> LeagueSettings:
    """Small 4-team league: 2 batters (OF) + 1 pitcher (P) = 3 rounds."""
    return LeagueSettings(
        name="Mock League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=0,
        roster_pitchers=1,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        positions={"OF": 2},
    )


def _mock_board() -> DraftBoard:
    """Board with enough players for 4 teams × 3 roster slots = 12 picks."""
    players: list[DraftBoardRow] = []
    # 8 batters (OF)
    batter_names = [
        "Mike Trout",
        "Juan Soto",
        "Aaron Judge",
        "Mookie Betts",
        "Shohei Ohtani",
        "Trea Turner",
        "Ronald Acuna",
        "Freddie Freeman",
    ]
    for i, name in enumerate(batter_names):
        players.append(
            DraftBoardRow(
                rank=i + 1,
                player_id=1000 + i,
                player_name=name,
                player_type=PlayerType.BATTER,
                position="OF",
                value=40.0 - i * 2.0,
                category_z_scores={"hr": 2.0 - i * 0.2},
                adp_overall=float(i + 1),
                adp_rank=i + 1,
                adp_delta=0,
            )
        )
    # 8 pitchers (SP, but fills "P" slot)
    pitcher_names = [
        "Gerrit Cole",
        "Spencer Strider",
        "Logan Webb",
        "Zack Wheeler",
        "Max Scherzer",
        "Justin Verlander",
        "Shane Bieber",
        "Dylan Cease",
    ]
    for i, name in enumerate(pitcher_names):
        players.append(
            DraftBoardRow(
                rank=9 + i,
                player_id=2000 + i,
                player_name=name,
                player_type=PlayerType.PITCHER,
                position="SP",
                value=30.0 - i * 2.0,
                category_z_scores={"w": 1.5 - i * 0.1},
                adp_overall=float(9 + i),
                adp_rank=9 + i,
                adp_delta=0,
            )
        )
    return DraftBoard(
        rows=players,
        batting_categories=("hr",),
        pitching_categories=("w",),
    )


def _patch_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch _fetch_draft_board_data to return small board/league."""
    monkeypatch.setattr(
        "fantasy_baseball_manager.cli.commands.mock_draft._fetch_draft_board_data",
        lambda *args, **kwargs: (_mock_board(), _mock_league()),
    )


class TestMockSingle:
    def test_prints_draft_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "single", "--season", "2026", "--seed", "42"],
        )
        assert result.exit_code == 0, result.output
        assert "Draft Log" in result.output
        assert "Mike Trout" in result.output

    def test_prints_roster_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "single", "--season", "2026", "--seed", "42"],
        )
        assert result.exit_code == 0, result.output
        assert "Roster" in result.output
        assert "Total" in result.output

    def test_strategy_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "single", "--season", "2026", "--strategy", "adp", "--seed", "1"],
        )
        assert result.exit_code == 0, result.output

    def test_position_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "single", "--season", "2026", "--position", "1", "--seed", "1"],
        )
        assert result.exit_code == 0, result.output

    def test_invalid_strategy_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "single", "--season", "2026", "--strategy", "invalid"],
        )
        assert result.exit_code != 0

    def test_respects_season_and_system(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args: list[tuple[object, ...]] = []

        def fake_fetch(*args: object, **kwargs: object) -> tuple[DraftBoard, LeagueSettings]:
            captured_args.append(args)
            return _mock_board(), _mock_league()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.mock_draft._fetch_draft_board_data",
            fake_fetch,
        )
        result = runner.invoke(
            app,
            ["draft", "mock", "single", "--season", "2025", "--system", "custom", "--seed", "1"],
        )
        assert result.exit_code == 0, result.output
        assert captured_args[0][0] == 2025  # season
        assert captured_args[0][1] == "custom"  # system

    def test_exclude_keepers_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args: list[tuple[object, ...]] = []

        def fake_fetch(*args: object, **kwargs: object) -> tuple[DraftBoard, LeagueSettings]:
            captured_args.append(args)
            return _mock_board(), _mock_league()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.mock_draft._fetch_draft_board_data",
            fake_fetch,
        )
        result = runner.invoke(
            app,
            ["draft", "mock", "single", "--season", "2026", "--exclude-keepers", "my-league", "--seed", "1"],
        )
        assert result.exit_code == 0, result.output
        assert captured_args[0][9] == "my-league"  # exclude_keepers


class TestMockBatch:
    def test_prints_summary_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "batch", "--season", "2026", "--simulations", "10", "--seed", "42"],
        )
        assert result.exit_code == 0, result.output
        assert "Avg" in result.output
        assert "Median" in result.output

    def test_prints_top_players(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "batch", "--season", "2026", "--simulations", "10", "--seed", "42"],
        )
        assert result.exit_code == 0, result.output
        assert "Player" in result.output
        assert "%" in result.output

    def test_prints_strategy_comparison(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "batch", "--season", "2026", "--simulations", "10", "--seed", "42"],
        )
        assert result.exit_code == 0, result.output
        assert "Win" in result.output

    def test_simulations_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            ["draft", "mock", "batch", "--season", "2026", "--simulations", "3", "--seed", "42"],
        )
        assert result.exit_code == 0, result.output
        assert "3" in result.output

    def test_exclude_keepers_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args: list[tuple[object, ...]] = []

        def fake_fetch(*args: object, **kwargs: object) -> tuple[DraftBoard, LeagueSettings]:
            captured_args.append(args)
            return _mock_board(), _mock_league()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.mock_draft._fetch_draft_board_data",
            fake_fetch,
        )
        result = runner.invoke(
            app,
            [
                "draft",
                "mock",
                "batch",
                "--season",
                "2026",
                "--exclude-keepers",
                "my-league",
                "--simulations",
                "3",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured_args[0][9] == "my-league"  # exclude_keepers


class TestMockCompare:
    def test_prints_comparison_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            [
                "draft",
                "mock",
                "compare",
                "--season",
                "2026",
                "--strategies",
                "adp,best-value",
                "--simulations",
                "10",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "adp" in result.output
        assert "best-value" in result.output

    def test_ranks_strategies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            [
                "draft",
                "mock",
                "compare",
                "--season",
                "2026",
                "--strategies",
                "best-value,random",
                "--simulations",
                "10",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "best-value" in result.output
        assert "random" in result.output

    def test_invalid_strategy_in_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_fetch(monkeypatch)
        result = runner.invoke(
            app,
            [
                "draft",
                "mock",
                "compare",
                "--season",
                "2026",
                "--strategies",
                "adp,bogus",
                "--simulations",
                "10",
            ],
        )
        assert result.exit_code != 0

    def test_respects_season_and_system(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args: list[tuple[object, ...]] = []

        def fake_fetch(*args: object, **kwargs: object) -> tuple[DraftBoard, LeagueSettings]:
            captured_args.append(args)
            return _mock_board(), _mock_league()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.mock_draft._fetch_draft_board_data",
            fake_fetch,
        )
        result = runner.invoke(
            app,
            [
                "draft",
                "mock",
                "compare",
                "--season",
                "2025",
                "--system",
                "custom",
                "--strategies",
                "adp,best-value",
                "--simulations",
                "10",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.output
        # compare calls _fetch once
        assert captured_args[0][0] == 2025  # season
        assert captured_args[0][1] == "custom"  # system

    def test_exclude_keepers_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args: list[tuple[object, ...]] = []

        def fake_fetch(*args: object, **kwargs: object) -> tuple[DraftBoard, LeagueSettings]:
            captured_args.append(args)
            return _mock_board(), _mock_league()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.mock_draft._fetch_draft_board_data",
            fake_fetch,
        )
        result = runner.invoke(
            app,
            [
                "draft",
                "mock",
                "compare",
                "--season",
                "2026",
                "--exclude-keepers",
                "my-league",
                "--strategies",
                "adp,best-value",
                "--simulations",
                "3",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured_args[0][9] == "my-league"  # exclude_keepers
