import sqlite3
from typing import Any

from fantasy_baseball_manager.domain.player import Team
from fantasy_baseball_manager.domain.result import Ok
from fantasy_baseball_manager.ingest.column_maps import (
    make_position_appearance_mapper,
    make_roster_stint_mapper,
)
from fantasy_baseball_manager.ingest.loader import Loader
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from tests.helpers import seed_player
from tests.ingest.conftest import FakeDataSource


def _seed_team(conn: sqlite3.Connection) -> int:
    repo = SqliteTeamRepo(conn)
    return repo.upsert(Team(abbreviation="LAA", name="Los Angeles Angels", league="AL", division="W"))


def _appearance_rows() -> list[dict[str, Any]]:
    return [
        {"playerID": "troutmi01", "yearID": 2023, "teamID": "LAA", "position": "CF", "games": 82},
        {"playerID": "troutmi01", "yearID": 2023, "teamID": "LAA", "position": "DH", "games": 25},
    ]


def _roster_rows() -> list[dict[str, Any]]:
    return [
        {"playerID": "troutmi01", "yearID": 2023, "teamID": "LAA"},
    ]


class TestPositionAppearanceLoader:
    def test_end_to_end(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, bbref_id="troutmi01", retro_id="troum001")
        players = SqlitePlayerRepo(conn).all()
        mapper = make_position_appearance_mapper(players)

        source = FakeDataSource(_appearance_rows())
        repo = SqlitePositionAppearanceRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "position_appearance", conn=conn)

        result = loader.load(season=2023)

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 2
        assert log.target_table == "position_appearance"

        appearances = repo.get_by_player_season(player_id, 2023)
        assert len(appearances) == 2
        positions = {a.position: a.games for a in appearances}
        assert positions["CF"] == 82
        assert positions["DH"] == 25

    def test_unknown_player_skipped(self, conn: sqlite3.Connection) -> None:
        seed_player(conn, bbref_id="troutmi01", retro_id="troum001")
        players = SqlitePlayerRepo(conn).all()
        mapper = make_position_appearance_mapper(players)

        rows = [{"playerID": "xxxxx999", "yearID": 2023, "teamID": "LAA", "position": "CF", "games": 50}]
        source = FakeDataSource(rows)
        repo = SqlitePositionAppearanceRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "position_appearance", conn=conn)

        result = loader.load(season=2023)

        assert isinstance(result, Ok)
        assert result.value.status == "success"
        assert result.value.rows_loaded == 0

    def test_upsert_idempotency(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, bbref_id="troutmi01", retro_id="troum001")
        players = SqlitePlayerRepo(conn).all()
        mapper = make_position_appearance_mapper(players)

        source = FakeDataSource(_appearance_rows())
        repo = SqlitePositionAppearanceRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "position_appearance", conn=conn)

        loader.load(season=2023)
        loader.load(season=2023)

        appearances = repo.get_by_player_season(player_id, 2023)
        assert len(appearances) == 2


class TestRosterStintLoader:
    def test_end_to_end(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, bbref_id="troutmi01", retro_id="troum001")
        team_id = _seed_team(conn)
        players = SqlitePlayerRepo(conn).all()
        teams = SqliteTeamRepo(conn).all()
        mapper = make_roster_stint_mapper(players, teams)

        source = FakeDataSource(_roster_rows())
        repo = SqliteRosterStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "roster_stint", conn=conn)

        result = loader.load(season=2023)

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.target_table == "roster_stint"

        stints = repo.get_by_player_season(player_id, 2023)
        assert len(stints) == 1
        assert stints[0].team_id == team_id
        assert stints[0].start_date == "2023-03-01"

    def test_unknown_team_skipped(self, conn: sqlite3.Connection) -> None:
        seed_player(conn, bbref_id="troutmi01", retro_id="troum001")
        _seed_team(conn)
        players = SqlitePlayerRepo(conn).all()
        teams = SqliteTeamRepo(conn).all()
        mapper = make_roster_stint_mapper(players, teams)

        rows = [{"playerID": "troutmi01", "yearID": 2023, "teamID": "NYY"}]
        source = FakeDataSource(rows)
        repo = SqliteRosterStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "roster_stint", conn=conn)

        result = loader.load(season=2023)

        assert isinstance(result, Ok)
        assert result.value.status == "success"
        assert result.value.rows_loaded == 0

    def test_upsert_idempotency(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, bbref_id="troutmi01", retro_id="troum001")
        _seed_team(conn)
        players = SqlitePlayerRepo(conn).all()
        teams = SqliteTeamRepo(conn).all()
        mapper = make_roster_stint_mapper(players, teams)

        source = FakeDataSource(_roster_rows())
        repo = SqliteRosterStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "roster_stint", conn=conn)

        loader.load(season=2023)
        loader.load(season=2023)

        stints = repo.get_by_player_season(player_id, 2023)
        assert len(stints) == 1
