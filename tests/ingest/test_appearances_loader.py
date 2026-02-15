import sqlite3

import pandas as pd

from fantasy_baseball_manager.domain.player import Player, Team
from fantasy_baseball_manager.ingest.column_maps import (
    make_position_appearance_mapper,
    make_roster_stint_mapper,
)
from fantasy_baseball_manager.ingest.loader import StatsLoader
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from tests.ingest.conftest import FakeDataSource


def _seed_player(conn: sqlite3.Connection) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, retro_id="troum001"))


def _seed_team(conn: sqlite3.Connection) -> int:
    repo = SqliteTeamRepo(conn)
    return repo.upsert(Team(abbreviation="LAA", name="Los Angeles Angels", league="AL", division="W"))


def _appearance_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"playerID": "troum001", "yearID": 2023, "teamID": "LAA", "position": "CF", "games": 82},
            {"playerID": "troum001", "yearID": 2023, "teamID": "LAA", "position": "DH", "games": 25},
        ]
    )


def _roster_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"playerID": "troum001", "yearID": 2023, "teamID": "LAA"},
        ]
    )


class TestPositionAppearanceLoader:
    def test_end_to_end(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_position_appearance_mapper(players)

        source = FakeDataSource(_appearance_df())
        repo = SqlitePositionAppearanceRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "position_appearance", conn=conn)

        log = loader.load(season=2023)

        assert log.status == "success"
        assert log.rows_loaded == 2
        assert log.target_table == "position_appearance"

        appearances = repo.get_by_player_season(player_id, 2023)
        assert len(appearances) == 2
        positions = {a.position: a.games for a in appearances}
        assert positions["CF"] == 82
        assert positions["DH"] == 25

    def test_unknown_player_skipped(self, conn: sqlite3.Connection) -> None:
        _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_position_appearance_mapper(players)

        df = pd.DataFrame([{"playerID": "xxxxx999", "yearID": 2023, "teamID": "LAA", "position": "CF", "games": 50}])
        source = FakeDataSource(df)
        repo = SqlitePositionAppearanceRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "position_appearance", conn=conn)

        log = loader.load(season=2023)

        assert log.status == "success"
        assert log.rows_loaded == 0

    def test_upsert_idempotency(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_position_appearance_mapper(players)

        source = FakeDataSource(_appearance_df())
        repo = SqlitePositionAppearanceRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "position_appearance", conn=conn)

        loader.load(season=2023)
        loader.load(season=2023)

        appearances = repo.get_by_player_season(player_id, 2023)
        assert len(appearances) == 2


class TestRosterStintLoader:
    def test_end_to_end(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        team_id = _seed_team(conn)
        players = SqlitePlayerRepo(conn).all()
        teams = SqliteTeamRepo(conn).all()
        mapper = make_roster_stint_mapper(players, teams)

        source = FakeDataSource(_roster_df())
        repo = SqliteRosterStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "roster_stint", conn=conn)

        log = loader.load(season=2023)

        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.target_table == "roster_stint"

        stints = repo.get_by_player_season(player_id, 2023)
        assert len(stints) == 1
        assert stints[0].team_id == team_id
        assert stints[0].start_date == "2023-03-01"

    def test_unknown_team_skipped(self, conn: sqlite3.Connection) -> None:
        _seed_player(conn)
        _seed_team(conn)
        players = SqlitePlayerRepo(conn).all()
        teams = SqliteTeamRepo(conn).all()
        mapper = make_roster_stint_mapper(players, teams)

        df = pd.DataFrame([{"playerID": "troum001", "yearID": 2023, "teamID": "NYY"}])
        source = FakeDataSource(df)
        repo = SqliteRosterStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "roster_stint", conn=conn)

        log = loader.load(season=2023)

        assert log.status == "success"
        assert log.rows_loaded == 0

    def test_upsert_idempotency(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        _seed_team(conn)
        players = SqlitePlayerRepo(conn).all()
        teams = SqliteTeamRepo(conn).all()
        mapper = make_roster_stint_mapper(players, teams)

        source = FakeDataSource(_roster_df())
        repo = SqliteRosterStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "roster_stint", conn=conn)

        loader.load(season=2023)
        loader.load(season=2023)

        stints = repo.get_by_player_season(player_id, 2023)
        assert len(stints) == 1
