from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo

if TYPE_CHECKING:
    import sqlite3

    import pytest

runner = CliRunner()


def _seed_valuation_data(
    conn: sqlite3.Connection,
    system: str = "zar",
    version: str = "1.0",
    player_type: PlayerType = PlayerType.BATTER,
) -> None:
    """Seed player and valuation data for valuations commands."""
    player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
    val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
    pid1 = player_repo.upsert(Player(name_first="Juan", name_last="Soto", mlbam_id=665742))
    pid2 = player_repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
    val_repo.upsert(
        Valuation(
            player_id=pid1,
            season=2025,
            system=system,
            version=version,
            projection_system="steamer",
            projection_version="2025.1",
            player_type=player_type,
            position="OF",
            value=42.5,
            rank=1,
            category_scores={"hr": 2.1, "sb": 0.5},
        )
    )
    val_repo.upsert(
        Valuation(
            player_id=pid2,
            season=2025,
            system=system,
            version=version,
            projection_system="steamer",
            projection_version="2025.1",
            player_type=player_type,
            position="DH",
            value=38.0,
            rank=2,
            category_scores={"hr": 2.8, "sb": -0.3},
        )
    )
    conn.commit()


class TestValuationsLookupCommand:
    def test_lookup_returns_breakdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "lookup", "Soto", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Juan Soto" in result.output
        assert "42.5" in result.output

    def test_lookup_system_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn, system="zar")
        _seed_valuation_data(db_conn, system="auction")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["valuations", "lookup", "Soto", "--season", "2025", "--system", "zar", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "zar" in result.output
        assert "auction" not in result.output

    def test_lookup_no_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "lookup", "Nobody", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No valuations found" in result.output


class TestValuationsRankingsCommand:
    def test_rankings_shows_leaderboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "rankings", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Juan Soto" in result.output
        assert "Aaron Judge" in result.output

    def test_rankings_filter_by_player_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_data(db_conn, player_type=PlayerType.BATTER)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["valuations", "rankings", "--season", "2025", "--player-type", "pitcher", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "No valuations found" in result.output

    def test_rankings_empty_season(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["valuations", "rankings", "--season", "2099", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No valuations found" in result.output


def _seed_valuation_eval_data(conn: sqlite3.Connection) -> None:
    """Seed predicted valuations + actual stats for evaluate command."""
    player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
    val_repo = SqliteValuationRepo(SingleConnectionProvider(conn))
    batting_repo = SqliteBattingStatsRepo(SingleConnectionProvider(conn))
    pitching_repo = SqlitePitchingStatsRepo(SingleConnectionProvider(conn))
    position_repo = SqlitePositionAppearanceRepo(SingleConnectionProvider(conn))

    pid1 = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    pid2 = player_repo.upsert(Player(name_first="Aaron", name_last="Judge", mlbam_id=592450))
    pid3 = player_repo.upsert(Player(name_first="Gerrit", name_last="Cole", mlbam_id=543037))

    # Predicted valuations
    val_repo.upsert(
        Valuation(
            player_id=pid1,
            season=2025,
            system="zar",
            version="production",
            projection_system="steamer",
            projection_version="v1",
            player_type=PlayerType.BATTER,
            position="of",
            value=40.0,
            rank=1,
            category_scores={"hr": 1.5, "r": 1.0},
        )
    )
    val_repo.upsert(
        Valuation(
            player_id=pid2,
            season=2025,
            system="zar",
            version="production",
            projection_system="steamer",
            projection_version="v1",
            player_type=PlayerType.BATTER,
            position="util",
            value=30.0,
            rank=2,
            category_scores={"hr": 0.8, "r": 0.5},
        )
    )
    val_repo.upsert(
        Valuation(
            player_id=pid3,
            season=2025,
            system="zar",
            version="production",
            projection_system="steamer",
            projection_version="v1",
            player_type=PlayerType.PITCHER,
            position="p",
            value=25.0,
            rank=3,
            category_scores={"w": 1.0, "sv": -0.5},
        )
    )

    # Actual stats
    batting_repo.upsert(BattingStats(player_id=pid1, season=2025, source="fangraphs", pa=600, hr=35, r=100))
    batting_repo.upsert(BattingStats(player_id=pid2, season=2025, source="fangraphs", pa=550, hr=45, r=110))
    pitching_repo.upsert(PitchingStats(player_id=pid3, season=2025, source="fangraphs", ip=200.0, w=15, sv=0))

    # Position appearances
    position_repo.upsert(PositionAppearance(player_id=pid1, season=2025, position="OF", games=150))
    position_repo.upsert(PositionAppearance(player_id=pid2, season=2025, position="OF", games=140))

    conn.commit()


def _eval_league() -> LeagueSettings:
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=2,
        budget=260,
        roster_batters=2,
        roster_pitchers=1,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="r", name="Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="Wins", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
            CategoryConfig(key="sv", name="Saves", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=1,
        positions={"of": 1},
    )


class TestValuationsEvaluateCommand:
    def test_valuations_evaluate_shows_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_valuation_eval_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.valuations.load_league", lambda name, path: _eval_league()
        )

        result = runner.invoke(
            app,
            ["valuations", "evaluate", "--season", "2025", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "MAE" in result.output

    def test_valuations_evaluate_no_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.valuations.load_league", lambda name, path: _eval_league()
        )

        result = runner.invoke(
            app,
            ["valuations", "evaluate", "--season", "2099", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "No matched players" in result.output

    def test_valuations_evaluate_help(self) -> None:
        result = runner.invoke(app, ["valuations", "evaluate", "--help"])
        assert result.exit_code == 0
        assert "evaluate" in result.output.lower() or "season" in result.output.lower()
