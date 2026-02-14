from datetime import datetime

import pandas as pd
import pytest

from fantasy_baseball_manager.ingest.column_maps import chadwick_row_to_player
from fantasy_baseball_manager.ingest.loader import PlayerLoader
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from tests.ingest.conftest import ErrorDataSource, FakeDataSource


def _chadwick_df(*rows: dict) -> pd.DataFrame:
    defaults = {
        "name_first": "Mike",
        "name_last": "Trout",
        "key_mlbam": 545361,
        "key_fangraphs": 10155,
        "key_bbref": "troutmi01",
        "key_retro": "troum001",
        "mlb_played_first": 2011.0,
        "mlb_played_last": 2024.0,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestPlayerLoader:
    def test_loads_players_and_writes_success_log(self, conn) -> None:
        source = FakeDataSource(_chadwick_df({}))
        player_repo = SqlitePlayerRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = PlayerLoader(source, player_repo, log_repo, chadwick_row_to_player)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.source_type == "test"
        assert log.source_detail == "fake"
        assert log.target_table == "player"
        assert log.error_message is None

        players = player_repo.all()
        assert len(players) == 1
        assert players[0].name_first == "Mike"
        assert players[0].mlbam_id == 545361

    def test_skips_rows_where_mapper_returns_none(self, conn) -> None:
        df = _chadwick_df(
            {},
            {"name_first": "Nobody", "key_mlbam": float("nan")},
        )
        source = FakeDataSource(df)
        player_repo = SqlitePlayerRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = PlayerLoader(source, player_repo, log_repo, chadwick_row_to_player)

        log = loader.load()

        assert log.rows_loaded == 1
        assert len(player_repo.all()) == 1

    def test_upsert_is_idempotent(self, conn) -> None:
        source = FakeDataSource(_chadwick_df({}))
        player_repo = SqlitePlayerRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = PlayerLoader(source, player_repo, log_repo, chadwick_row_to_player)

        loader.load()
        loader.load()

        assert len(player_repo.all()) == 1

    def test_error_during_fetch_writes_error_log_and_reraises(self, conn) -> None:
        source = ErrorDataSource()
        player_repo = SqlitePlayerRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = PlayerLoader(source, player_repo, log_repo, chadwick_row_to_player)

        with pytest.raises(RuntimeError, match="fetch failed"):
            loader.load()

        logs = log_repo.get_by_target_table("player")
        assert len(logs) == 1
        assert logs[0].status == "error"
        assert logs[0].rows_loaded == 0
        assert "fetch failed" in (logs[0].error_message or "")

    def test_timestamps_are_valid_iso_strings(self, conn) -> None:
        source = FakeDataSource(_chadwick_df({}))
        player_repo = SqlitePlayerRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = PlayerLoader(source, player_repo, log_repo, chadwick_row_to_player)

        log = loader.load()

        started = datetime.fromisoformat(log.started_at)
        finished = datetime.fromisoformat(log.finished_at)
        assert started <= finished

    def test_empty_dataframe_loads_zero_rows(self, conn) -> None:
        source = FakeDataSource(pd.DataFrame())
        player_repo = SqlitePlayerRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = PlayerLoader(source, player_repo, log_repo, chadwick_row_to_player)

        log = loader.load()

        assert log.status == "success"
        assert log.rows_loaded == 0
        assert len(player_repo.all()) == 0

    def test_multiple_players(self, conn) -> None:
        df = _chadwick_df(
            {
                "name_first": "Mike",
                "name_last": "Trout",
                "key_mlbam": 545361,
                "key_fangraphs": 10155,
                "key_bbref": "troutmi01",
                "key_retro": "troum001",
            },
            {
                "name_first": "Shohei",
                "name_last": "Ohtani",
                "key_mlbam": 660271,
                "key_fangraphs": 19755,
                "key_bbref": "ohtansh01",
                "key_retro": "ohtas001",
            },
        )
        source = FakeDataSource(df)
        player_repo = SqlitePlayerRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = PlayerLoader(source, player_repo, log_repo, chadwick_row_to_player)

        log = loader.load()

        assert log.rows_loaded == 2
        assert len(player_repo.all()) == 2
