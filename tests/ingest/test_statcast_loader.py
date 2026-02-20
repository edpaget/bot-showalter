import sqlite3
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.db.connection import attach_database, create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.domain.statcast_pitch import StatcastPitch
from fantasy_baseball_manager.ingest.column_maps import statcast_pitch_mapper
from fantasy_baseball_manager.ingest.loader import Loader
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.statcast_pitch_repo import SqliteStatcastPitchRepo
from tests.ingest.conftest import ErrorDataSource, FakeDataSource


def _statcast_rows(*overrides: dict[str, Any]) -> list[dict[str, Any]]:
    defaults: dict[str, Any] = {
        "game_pk": 718001,
        "game_date": "2024-06-15",
        "batter": 545361,
        "pitcher": 477132,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitch_type": "FF",
        "release_speed": 95.2,
        "release_spin_rate": 2400.0,
        "pfx_x": -5.1,
        "pfx_z": 10.3,
        "plate_x": 0.5,
        "plate_z": 2.8,
        "zone": 5,
        "events": "single",
        "description": "hit_into_play",
        "launch_speed": 102.3,
        "launch_angle": 15.0,
        "hit_distance_sc": 250.0,
        "barrel": 1,
        "estimated_ba_using_speedangle": 0.620,
        "estimated_woba_using_speedangle": 0.850,
        "estimated_slg_using_speedangle": 0.750,
    }
    return [{**defaults, **o} for o in overrides]


class TestStatcastLoaderIntegration:
    def test_loads_statcast_pitches_via_stats_loader(
        self, statcast_conn: sqlite3.Connection, conn: sqlite3.Connection
    ) -> None:
        rows = _statcast_rows({})
        source = FakeDataSource(rows)
        pitch_repo = SqliteStatcastPitchRepo(statcast_conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(
            source, pitch_repo, log_repo, statcast_pitch_mapper, "statcast_pitch", conn=statcast_conn, log_conn=conn
        )

        result = loader.load()

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 1
        assert log.target_table == "statcast_pitch"

        results = pitch_repo.get_by_game(718001)
        assert len(results) == 1
        assert results[0].pitch_type == "FF"
        assert results[0].batter_id == 545361

    def test_skips_rows_with_nan_batter(self, statcast_conn: sqlite3.Connection, conn: sqlite3.Connection) -> None:
        rows = _statcast_rows(
            {},
            {"batter": float("nan"), "at_bat_number": 2, "pitch_number": 1},
        )
        source = FakeDataSource(rows)
        pitch_repo = SqliteStatcastPitchRepo(statcast_conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(
            source, pitch_repo, log_repo, statcast_pitch_mapper, "statcast_pitch", conn=statcast_conn, log_conn=conn
        )

        result = loader.load()

        assert isinstance(result, Ok)
        assert result.value.rows_loaded == 1

    def test_skips_rows_with_missing_required_fields(
        self, statcast_conn: sqlite3.Connection, conn: sqlite3.Connection
    ) -> None:
        rows = _statcast_rows(
            {},
            {"game_pk": float("nan"), "at_bat_number": 2, "pitch_number": 1},
        )
        source = FakeDataSource(rows)
        pitch_repo = SqliteStatcastPitchRepo(statcast_conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(
            source, pitch_repo, log_repo, statcast_pitch_mapper, "statcast_pitch", conn=statcast_conn, log_conn=conn
        )

        result = loader.load()

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 1
        assert pitch_repo.count() == 1

    def test_empty_list_loads_zero_rows(self, statcast_conn: sqlite3.Connection, conn: sqlite3.Connection) -> None:
        source = FakeDataSource([])
        pitch_repo = SqliteStatcastPitchRepo(statcast_conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(
            source, pitch_repo, log_repo, statcast_pitch_mapper, "statcast_pitch", conn=statcast_conn, log_conn=conn
        )

        result = loader.load()

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 0
        assert pitch_repo.count() == 0

    def test_fetch_error_writes_error_log(self, statcast_conn: sqlite3.Connection, conn: sqlite3.Connection) -> None:
        source = ErrorDataSource()
        pitch_repo = SqliteStatcastPitchRepo(statcast_conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(
            source, pitch_repo, log_repo, statcast_pitch_mapper, "statcast_pitch", conn=statcast_conn, log_conn=conn
        )

        result = loader.load()

        assert isinstance(result, Err)
        assert "fetch failed" in result.error.message

        logs = log_repo.get_by_target_table("statcast_pitch")
        assert len(logs) == 1
        assert logs[0].status == "error"
        assert logs[0].rows_loaded == 0
        assert "fetch failed" in (logs[0].error_message or "")

    def test_upsert_deduplicates_on_reload(self, statcast_conn: sqlite3.Connection, conn: sqlite3.Connection) -> None:
        rows = _statcast_rows({})
        source = FakeDataSource(rows)
        pitch_repo = SqliteStatcastPitchRepo(statcast_conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = Loader(
            source, pitch_repo, log_repo, statcast_pitch_mapper, "statcast_pitch", conn=statcast_conn, log_conn=conn
        )

        loader.load()
        loader.load()

        assert pitch_repo.count() == 1


class TestAttachJoin:
    def test_attach_join_statcast_to_player(self, tmp_path: Path) -> None:
        stats_path = tmp_path / "stats.db"
        statcast_path = tmp_path / "statcast.db"

        # Create stats.db and seed a player
        stats_conn = create_connection(stats_path)
        player_repo = SqlitePlayerRepo(stats_conn)
        player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        stats_conn.commit()
        stats_conn.close()

        # Create statcast.db and insert a pitch
        sc_conn = create_statcast_connection(statcast_path)
        pitch_repo = SqliteStatcastPitchRepo(sc_conn)
        pitch_repo.upsert(
            StatcastPitch(
                game_pk=718001,
                game_date="2024-06-15",
                batter_id=545361,
                pitcher_id=477132,
                at_bat_number=1,
                pitch_number=1,
                pitch_type="FF",
            )
        )
        sc_conn.commit()
        sc_conn.close()

        # Re-open stats and ATTACH statcast
        stats_conn = create_connection(stats_path)
        attach_database(stats_conn, statcast_path, "statcast")

        rows = stats_conn.execute("""SELECT p.name_first, p.name_last, sc.pitch_type
               FROM player p
               JOIN statcast.statcast_pitch sc ON p.mlbam_id = sc.batter_id
               WHERE sc.game_pk = 718001""").fetchall()

        assert len(rows) == 1
        assert tuple(rows[0]) == ("Mike", "Trout", "FF")
        stats_conn.close()
