from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.player import Player, Team
from fantasy_baseball_manager.domain.roster_stint import RosterStint
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo

if TYPE_CHECKING:
    import sqlite3

    import pytest

runner = CliRunner()


def _seed_bio_data(conn: sqlite3.Connection) -> None:
    """Seed two teams (NYY, NYM) with one player each on 2025 rosters."""
    team_repo = SqliteTeamRepo(conn)
    player_repo = SqlitePlayerRepo(conn)
    stint_repo = SqliteRosterStintRepo(conn)

    nyy_id = team_repo.upsert(Team(abbreviation="NYY", name="New York Yankees", league="AL", division="East"))
    nym_id = team_repo.upsert(Team(abbreviation="NYM", name="New York Mets", league="NL", division="East"))

    judge_id = player_repo.upsert(
        Player(name_first="Aaron", name_last="Judge", mlbam_id=592450, birth_date="1992-04-26")
    )
    alonso_id = player_repo.upsert(
        Player(name_first="Pete", name_last="Alonso", mlbam_id=624413, birth_date="1994-12-07")
    )

    stint_repo.upsert(RosterStint(player_id=judge_id, team_id=nyy_id, season=2025, start_date="2025-03-27"))
    stint_repo.upsert(RosterStint(player_id=alonso_id, team_id=nym_id, season=2025, start_date="2025-03-27"))
    conn.commit()


class TestBioCommand:
    def test_nickname_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_bio_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["bio", "--team", "Yankees", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Resolved 'Yankees' to NYY" in result.output
        assert "Aaron Judge" in result.output

    def test_ambiguous_city(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_bio_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["bio", "--team", "New York", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 1, result.output
        assert "matches multiple teams" in result.output
        assert "NYY" in result.output
        assert "NYM" in result.output

    def test_no_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_bio_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["bio", "--team", "xyzabc", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 1, result.output
        assert "No team found" in result.output

    def test_exact_abbreviation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_bio_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["bio", "--team", "NYY", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Resolved" not in result.output
        assert "Aaron Judge" in result.output

    def test_no_team_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_bio_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["bio", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Aaron Judge" in result.output
        assert "Pete Alonso" in result.output
