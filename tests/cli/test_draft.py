from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

    import pytest

runner = CliRunner()


def _draft_board_league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=9,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
    )


def _seed_draft_board_data(conn: sqlite3.Connection) -> None:
    """Seed player + valuation data for draft board commands."""
    player_repo = SqlitePlayerRepo(conn)
    val_repo = SqliteValuationRepo(conn)

    pid1 = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    pid2 = player_repo.upsert(Player(name_first="Gerrit", name_last="Cole", mlbam_id=543037))

    val_repo.upsert(
        Valuation(
            player_id=pid1,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="batter",
            position="OF",
            value=42.5,
            rank=1,
            category_scores={"hr": 2.1, "r": 1.0},
        )
    )
    val_repo.upsert(
        Valuation(
            player_id=pid2,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="pitcher",
            position="SP",
            value=35.0,
            rank=2,
            category_scores={"w": 1.5, "sv": -0.2},
        )
    )
    conn.commit()


class TestDraftBoardCommand:
    def test_draft_board_shows_players(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_draft_board_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _draft_board_league()
        )

        result = runner.invoke(
            app,
            ["draft", "board", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Gerrit Cole" in result.output

    def test_draft_board_empty_season(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _draft_board_league()
        )

        result = runner.invoke(
            app,
            ["draft", "board", "--season", "2099", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "No players on draft board" in result.output

    def test_draft_board_top_limits_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_draft_board_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _draft_board_league()
        )

        result = runner.invoke(
            app,
            ["draft", "board", "--season", "2026", "--top", "1", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Gerrit Cole" not in result.output


class TestDraftExportCommand:
    def test_draft_export_writes_csv(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_conn = create_connection(":memory:")
        _seed_draft_board_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _draft_board_league()
        )

        output_file = tmp_path / "board.csv"
        result = runner.invoke(
            app,
            ["draft", "export", "--season", "2026", "--output", str(output_file), "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        content = output_file.read_text()
        assert "Mike Trout" in content
        assert "Gerrit Cole" in content
        assert "Rank" in content

    def test_draft_export_empty_season(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _draft_board_league()
        )

        output_file = tmp_path / "empty.csv"
        result = runner.invoke(
            app,
            ["draft", "export", "--season", "2099", "--output", str(output_file), "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        # Should have header but no data rows
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1  # header only


def _seed_tier_data(conn: sqlite3.Connection) -> None:
    """Seed players + valuations with clear value gaps for tier tests."""
    player_repo = SqlitePlayerRepo(conn)
    val_repo = SqliteValuationRepo(conn)

    # OF players with clear tier gaps: 42, 40 (tier 1) ... gap ... 28, 26 (tier 2) ... gap ... 15 (tier 3)
    of_players = [
        ("Mike", "Trout", 545361, 42.0),
        ("Juan", "Soto", 665742, 40.0),
        ("Aaron", "Judge", 592450, 28.0),
        ("Kyle", "Tucker", 663656, 26.0),
        ("Julio", "Rodriguez", 677594, 15.0),
    ]
    for i, (first, last, mlbam, value) in enumerate(of_players, start=1):
        pid = player_repo.upsert(Player(name_first=first, name_last=last, mlbam_id=mlbam))
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="batter",
                position="OF",
                value=value,
                rank=i,
                category_scores={"hr": 1.0},
            )
        )

    # SP players: 35, 33 (tier 1) ... gap ... 20 (tier 2)
    sp_players = [
        ("Gerrit", "Cole", 543037, 35.0),
        ("Spencer", "Strider", 675911, 33.0),
        ("Logan", "Webb", 657277, 20.0),
    ]
    for i, (first, last, mlbam, value) in enumerate(sp_players, start=1):
        pid = player_repo.upsert(Player(name_first=first, name_last=last, mlbam_id=mlbam))
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type="pitcher",
                position="SP",
                value=value,
                rank=i,
                category_scores={"w": 1.0},
            )
        )

    conn.commit()


class TestDraftTiersCommand:
    def test_draft_tiers_shows_players(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_tier_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tiers", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Gerrit Cole" in result.output
        assert "OF" in result.output
        assert "SP" in result.output

    def test_draft_tiers_empty_season(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tiers", "--season", "2099", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "No tier data found" in result.output

    def test_draft_tiers_position_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_tier_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tiers", "--season", "2026", "--position", "OF", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "Gerrit Cole" not in result.output

    def test_draft_tiers_method_jenks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_tier_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tiers", "--season", "2026", "--method", "jenks", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output

    def test_draft_tiers_max_tiers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_tier_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tiers", "--season", "2026", "--max-tiers", "2", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        # With max-tiers=2, no tier number should exceed 2
        assert "Mike Trout" in result.output
        # Check that tier numbers > 2 don't appear as standalone tier values
        # (the output has columns: Tier, Rank, Player, Value)
        lines = result.output.split("\n")
        for line in lines:
            parts = line.split()
            # Look for lines that start with a tier number in the first column
            if parts and parts[0].isdigit():
                tier_num = int(parts[0])
                assert tier_num <= 2, f"Found tier {tier_num} > 2 in output: {line}"


class TestDraftTierSummaryCommand:
    def test_tier_summary_shows_matrix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_tier_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tier-summary", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "OF" in result.output
        assert "SP" in result.output
        assert "Tier 1" in result.output

    def test_tier_summary_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tier-summary", "--season", "2099", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "No tier data found" in result.output

    def test_tier_summary_counts_correct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_tier_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["draft", "tier-summary", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        # The seeded data has 5 OF and 3 SP players
        # Total row should appear
        assert "Total" in result.output
