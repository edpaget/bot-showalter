import sqlite3

from fantasy_baseball_manager.db.connection import get_schema_version
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection


class TestCreateStatcastConnection:
    def test_returns_connection(self) -> None:
        conn = create_statcast_connection(":memory:")
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_statcast_pitch_table_exists(self) -> None:
        conn = create_statcast_connection(":memory:")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "statcast_pitch" in tables
        conn.close()

    def test_statcast_pitch_columns(self) -> None:
        conn = create_statcast_connection(":memory:")
        cursor = conn.execute("PRAGMA table_info(statcast_pitch)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "game_pk",
            "game_date",
            "batter_id",
            "pitcher_id",
            "at_bat_number",
            "pitch_number",
            "pitch_type",
            "release_speed",
            "release_spin_rate",
            "pfx_x",
            "pfx_z",
            "plate_x",
            "plate_z",
            "zone",
            "events",
            "description",
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "barrel",
            "estimated_ba_using_speedangle",
            "estimated_woba_using_speedangle",
            "loaded_at",
        }
        assert expected.issubset(columns)
        conn.close()

    def test_indexes_exist(self) -> None:
        conn = create_statcast_connection(":memory:")
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "idx_sc_pitcher_date" in indexes
        assert "idx_sc_batter_date" in indexes
        assert "idx_sc_game" in indexes
        conn.close()

    def test_unique_constraint_upserts(self) -> None:
        conn = create_statcast_connection(":memory:")
        conn.execute(
            "INSERT INTO statcast_pitch (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, pitch_type)"
            " VALUES (718001, '2024-06-15', 545361, 477132, 1, 1, 'FF')"
        )
        conn.execute(
            "INSERT INTO statcast_pitch (game_pk, game_date, batter_id, pitcher_id, at_bat_number, pitch_number, pitch_type)"
            " VALUES (718001, '2024-06-15', 545361, 477132, 1, 1, 'SL')"
            " ON CONFLICT(game_pk, at_bat_number, pitch_number) DO UPDATE SET pitch_type=excluded.pitch_type"
        )
        conn.commit()
        rows = conn.execute("SELECT pitch_type FROM statcast_pitch").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "SL"
        conn.close()

    def test_schema_version_is_set(self) -> None:
        conn = create_statcast_connection(":memory:")
        assert get_schema_version(conn) == 1
        conn.close()

    def test_stats_tables_absent(self) -> None:
        conn = create_statcast_connection(":memory:")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "player" not in tables
        assert "batting_stats" not in tables
        assert "pitching_stats" not in tables
        conn.close()
