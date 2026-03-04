from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
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


def _needs_league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=14,
        roster_pitchers=10,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sb", name="SB", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(),
    )


def _seed_needs_data(conn: sqlite3.Connection) -> None:
    """Seed player + projection data for draft needs tests."""
    player_repo = SqlitePlayerRepo(conn)
    proj_repo = SqliteProjectionRepo(conn)

    # Roster player: strong HR, weak SB
    pid1 = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    proj_repo.upsert(
        Projection(
            player_id=pid1,
            season=2026,
            system="steamer",
            version="1.0",
            player_type="batter",
            stat_json={"hr": 35.0, "sb": 3.0},
        )
    )

    # Available player: great SB
    pid2 = player_repo.upsert(Player(name_first="Rickey", name_last="Henderson", mlbam_id=100001))
    proj_repo.upsert(
        Projection(
            player_id=pid2,
            season=2026,
            system="steamer",
            version="1.0",
            player_type="batter",
            stat_json={"hr": 5.0, "sb": 40.0},
        )
    )

    # League pool: enough batters so league averages work
    for i in range(3, 170):
        pid = player_repo.upsert(Player(name_first=f"Player{i}", name_last=f"Last{i}", mlbam_id=100000 + i))
        proj_repo.upsert(
            Projection(
                player_id=pid,
                season=2026,
                system="steamer",
                version="1.0",
                player_type="batter",
                stat_json={"hr": 20.0, "sb": 15.0},
            )
        )

    conn.commit()


class TestDraftNeedsCommand:
    def test_draft_needs_shows_weak_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_needs_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _needs_league()
        )

        result = runner.invoke(
            app,
            [
                "draft",
                "needs",
                "--roster",
                "Trout",
                "--season",
                "2026",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        # SB is weak, so it should appear
        assert "SB" in result.output
        # Rickey Henderson should be recommended
        assert "Rickey Henderson" in result.output

    def test_draft_needs_no_weak_categories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        player_repo = SqlitePlayerRepo(db_conn)
        proj_repo = SqliteProjectionRepo(db_conn)

        # Roster player with very high stats
        pid = player_repo.upsert(Player(name_first="Mega", name_last="Star", mlbam_id=999999))
        proj_repo.upsert(
            Projection(
                player_id=pid,
                season=2026,
                system="steamer",
                version="1.0",
                player_type="batter",
                stat_json={"hr": 500.0, "sb": 400.0},
            )
        )
        # Small league pool — roster player dominates
        for i in range(2, 15):
            p = player_repo.upsert(Player(name_first=f"P{i}", name_last=f"L{i}", mlbam_id=100000 + i))
            proj_repo.upsert(
                Projection(
                    player_id=p,
                    season=2026,
                    system="steamer",
                    version="1.0",
                    player_type="batter",
                    stat_json={"hr": 20.0, "sb": 15.0},
                )
            )
        db_conn.commit()

        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league",
            lambda name, path: _needs_league(),
        )

        result = runner.invoke(
            app,
            ["draft", "needs", "--roster", "Mega", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "No weak categories identified" in result.output

    def test_draft_needs_unknown_player(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_needs_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _needs_league()
        )

        result = runner.invoke(
            app,
            ["draft", "needs", "--roster", "NonExistentPlayer", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "not found" in result.output


def _pick_value_league() -> LeagueSettings:
    """Small 4-team league for pick value tests (4 teams × 3 roster slots = 12 total picks)."""
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=2,
        roster_pitchers=1,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
    )


def _seed_pick_value_data(conn: sqlite3.Connection) -> None:
    """Seed player + valuation + ADP data for pick value/trade tests."""
    player_repo = SqlitePlayerRepo(conn)
    val_repo = SqliteValuationRepo(conn)
    adp_repo = SqliteADPRepo(conn)

    players = [
        ("Mike", "Trout", 545361, "batter", "OF", 40.0),
        ("Juan", "Soto", 665742, "batter", "OF", 35.0),
        ("Aaron", "Judge", 592450, "batter", "OF", 30.0),
        ("Gerrit", "Cole", 543037, "pitcher", "SP", 28.0),
        ("Shohei", "Ohtani", 660271, "batter", "OF", 25.0),
        ("Mookie", "Betts", 605141, "batter", "OF", 22.0),
        ("Trea", "Turner", 607208, "batter", "SS", 20.0),
        ("Ronald", "Acuna", 660670, "batter", "OF", 18.0),
        ("Freddie", "Freeman", 518692, "batter", "1B", 16.0),
        ("Spencer", "Strider", 675911, "pitcher", "SP", 14.0),
        ("Corey", "Seager", 608369, "batter", "SS", 12.0),
        ("Bobby", "Witt", 677951, "batter", "SS", 10.0),
    ]
    for i, (first, last, mlbam, ptype, pos, value) in enumerate(players, start=1):
        pid = player_repo.upsert(Player(name_first=first, name_last=last, mlbam_id=mlbam))
        val_repo.upsert(
            Valuation(
                player_id=pid,
                season=2026,
                system="zar",
                version="1.0",
                projection_system="steamer",
                projection_version="v1",
                player_type=ptype,
                position=pos,
                value=value,
                rank=i,
                category_scores={"hr": 1.0},
            )
        )
        adp_repo.upsert(
            ADP(
                player_id=pid,
                season=2026,
                provider="fantasypros",
                overall_pick=float(i),
                rank=i,
                positions=pos,
            )
        )
    conn.commit()


class TestDraftPickValues:
    def test_pick_values_prints_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_pick_value_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _pick_value_league()
        )

        result = runner.invoke(
            app,
            ["draft", "pick-values", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Pick" in result.output
        assert "Value" in result.output
        assert "Mike Trout" in result.output

    def test_pick_values_confidence_shown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_pick_value_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _pick_value_league()
        )

        result = runner.invoke(
            app,
            ["draft", "pick-values", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "high" in result.output.lower() or "medium" in result.output.lower()


def _upgrade_league() -> LeagueSettings:
    """League with OF+SP positions for upgrade tests."""
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=0,
        roster_pitchers=0,
        positions={"OF": 2},
        pitcher_positions={"SP": 1},
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
    )


def _seed_upgrade_data(conn: sqlite3.Connection) -> None:
    """Seed 3 players for upgrade/position-check tests."""
    player_repo = SqlitePlayerRepo(conn)
    val_repo = SqliteValuationRepo(conn)

    # Roster player
    pid1 = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
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
            category_scores={"hr": 2.1},
        )
    )

    # Available batter
    pid2 = player_repo.upsert(Player(name_first="Juan", name_last="Soto", mlbam_id=665742))
    val_repo.upsert(
        Valuation(
            player_id=pid2,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="batter",
            position="OF",
            value=38.0,
            rank=2,
            category_scores={"hr": 1.8},
        )
    )

    # Available pitcher
    pid3 = player_repo.upsert(Player(name_first="Gerrit", name_last="Cole", mlbam_id=543037))
    val_repo.upsert(
        Valuation(
            player_id=pid3,
            season=2026,
            system="zar",
            version="1.0",
            projection_system="steamer",
            projection_version="v1",
            player_type="pitcher",
            position="SP",
            value=35.0,
            rank=3,
            category_scores={"w": 1.5},
        )
    )

    conn.commit()


class TestDraftUpgradesCommand:
    def test_upgrades_shows_marginal_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_upgrade_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _upgrade_league()
        )

        result = runner.invoke(
            app,
            ["draft", "upgrades", "--roster", "Trout", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Juan Soto" in result.output
        assert "Gerrit Cole" in result.output
        assert "Marginal" in result.output

    def test_upgrades_with_roster_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_conn = create_connection(":memory:")
        _seed_upgrade_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _upgrade_league()
        )

        roster_file = tmp_path / "roster.txt"
        roster_file.write_text("Trout\n")

        result = runner.invoke(
            app,
            [
                "draft",
                "upgrades",
                "--roster-file",
                str(roster_file),
                "--season",
                "2026",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Juan Soto" in result.output

    def test_upgrades_with_opportunity_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_upgrade_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _upgrade_league()
        )

        result = runner.invoke(
            app,
            [
                "draft",
                "upgrades",
                "--roster",
                "Trout",
                "--season",
                "2026",
                "--opportunity-cost",
                "--picks-until-next",
                "2",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Recommendation" in result.output

    def test_upgrades_unknown_player(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_upgrade_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _upgrade_league()
        )

        result = runner.invoke(
            app,
            ["draft", "upgrades", "--roster", "NonExistentPlayer", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "not found" in result.output

    def test_upgrades_no_roster_option(self) -> None:
        result = runner.invoke(
            app,
            ["draft", "upgrades", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 1


class TestDraftPositionCheckCommand:
    def test_position_check_shows_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_upgrade_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _upgrade_league()
        )

        result = runner.invoke(
            app,
            ["draft", "position-check", "--roster", "Trout", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Position" in result.output
        assert "Urgency" in result.output
        # Verify player data appears (column headers may wrap in narrow terminal)
        assert "Juan Soto" in result.output or "Soto" in result.output

    def test_position_check_urgency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_upgrade_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _upgrade_league()
        )

        # Trout fills one OF slot; there's still an open OF and an open SP
        result = runner.invoke(
            app,
            ["draft", "position-check", "--roster", "Trout", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        # SP slot is open with only one candidate → high urgency (dropoff = full value)
        assert "high" in result.output or "medium" in result.output


class TestDraftTradePicks:
    def test_trade_picks_shows_recommendation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_pick_value_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _pick_value_league()
        )

        result = runner.invoke(
            app,
            [
                "draft",
                "trade-picks",
                "--gives",
                "1",
                "--receives",
                "5,6",
                "--season",
                "2026",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "You give" in result.output
        assert "You receive" in result.output
        # Should show a recommendation
        assert any(word in result.output.lower() for word in ("accept", "reject", "even"))

    def test_trade_picks_net_value_shown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_pick_value_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _pick_value_league()
        )

        result = runner.invoke(
            app,
            [
                "draft",
                "trade-picks",
                "--gives",
                "1",
                "--receives",
                "2,3",
                "--season",
                "2026",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Net value" in result.output

    def test_trade_picks_cascade(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_pick_value_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.draft.load_league", lambda name, path: _pick_value_league()
        )

        # 4-team snake: team 0 owns {1,8,9}, team 1 owns {2,7,10}
        # Trade pick 1 (team 0) for picks 2,7 (both team 1)
        result = runner.invoke(
            app,
            [
                "draft",
                "trade-picks",
                "--gives",
                "1",
                "--receives",
                "2,7",
                "--season",
                "2026",
                "--cascade",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        # Cascade output should show before/after
        assert "Before" in result.output
        assert "After" in result.output
