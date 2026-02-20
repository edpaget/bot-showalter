import sqlite3
from typing import Any

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.result import Ok
from fantasy_baseball_manager.ingest.column_maps import make_milb_batting_mapper
from fantasy_baseball_manager.ingest.loader import Loader
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.minor_league_batting_stats_repo import SqliteMinorLeagueBattingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from tests.ingest.conftest import FakeDataSource


def _seed_player(conn: sqlite3.Connection, *, mlbam_id: int = 545361) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=mlbam_id))


def _milb_rows(*overrides: dict[str, Any]) -> list[dict[str, Any]]:
    defaults: dict[str, Any] = {
        "mlbam_id": 545361,
        "season": 2024,
        "level": "AAA",
        "league": "International League",
        "team": "Syracuse Mets",
        "g": 120,
        "pa": 500,
        "ab": 450,
        "h": 130,
        "doubles": 25,
        "triples": 3,
        "hr": 18,
        "r": 70,
        "rbi": 65,
        "bb": 40,
        "so": 100,
        "sb": 15,
        "cs": 5,
        "avg": 0.289,
        "obp": 0.350,
        "slg": 0.480,
        "age": 24.5,
        "hbp": 8,
        "sf": 4,
        "sh": 1,
    }
    return [{**defaults, **o} for o in (overrides or [{}])]


class TestMilbLoader:
    def test_end_to_end_load(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_milb_batting_mapper(players)

        source = FakeDataSource(_milb_rows())
        repo = SqliteMinorLeagueBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "minor_league_batting_stats", conn=conn)

        result = loader.load(season=2024, level="AAA")

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.target_table == "minor_league_batting_stats"

        results = repo.get_by_player(player_id)
        assert len(results) == 1
        assert results[0].hr == 18
        assert results[0].level == "AAA"

    def test_unknown_players_skipped(self, conn: sqlite3.Connection) -> None:
        _seed_player(conn, mlbam_id=545361)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_milb_batting_mapper(players)

        rows = _milb_rows({"mlbam_id": 999999})
        source = FakeDataSource(rows)
        repo = SqliteMinorLeagueBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "minor_league_batting_stats", conn=conn)

        result = loader.load(season=2024, level="AAA")

        assert isinstance(result, Ok)
        assert result.value.status == "success"
        assert result.value.rows_loaded == 0

    def test_multi_level_season(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_milb_batting_mapper(players)

        rows = _milb_rows(
            {"level": "AA", "team": "Binghamton Rumble Ponies"},
            {"level": "AAA", "team": "Syracuse Mets"},
        )
        source = FakeDataSource(rows)
        repo = SqliteMinorLeagueBattingStatsRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(source, repo, log_repo, mapper, "minor_league_batting_stats", conn=conn)

        result = loader.load(season=2024, level="AAA")

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 2

        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 2
        levels = {r.level for r in results}
        assert levels == {"AA", "AAA"}
