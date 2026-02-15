import sqlite3
from pathlib import Path

import pytest

from fantasy_baseball_manager.db.connection import (
    attach_database,
    create_connection,
    get_schema_version,
)


class TestCreateConnection:
    def test_returns_connection(self, tmp_path: object) -> None:

        db_path = Path(str(tmp_path)) / "test.db"
        conn = create_connection(db_path)
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_enables_wal_mode(self, tmp_path: object) -> None:

        db_path = Path(str(tmp_path)) / "test.db"
        conn = create_connection(db_path)
        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result is not None
        assert result[0] == "wal"
        conn.close()

    def test_enables_foreign_keys(self, tmp_path: object) -> None:

        db_path = Path(str(tmp_path)) / "test.db"
        conn = create_connection(db_path)
        result = conn.execute("PRAGMA foreign_keys").fetchone()
        assert result is not None
        assert result[0] == 1
        conn.close()

    def test_creates_all_tables(self, tmp_path: object) -> None:

        db_path = Path(str(tmp_path)) / "test.db"
        conn = create_connection(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        expected = {
            "schema_version",
            "player",
            "team",
            "batting_stats",
            "pitching_stats",
            "projection",
            "feature_set",
            "dataset",
            "model_run",
            "load_log",
        }
        assert expected.issubset(tables)
        conn.close()

    def test_schema_version_is_set(self, tmp_path: object) -> None:

        db_path = Path(str(tmp_path)) / "test.db"
        conn = create_connection(db_path)
        assert get_schema_version(conn) >= 1
        conn.close()

    def test_idempotent_reopen(self, tmp_path: object) -> None:

        db_path = Path(str(tmp_path)) / "test.db"
        conn1 = create_connection(db_path)
        conn1.close()
        conn2 = create_connection(db_path)
        assert get_schema_version(conn2) >= 1
        conn2.close()

    def test_in_memory_connection(self) -> None:
        conn = create_connection(":memory:")
        assert isinstance(conn, sqlite3.Connection)
        assert get_schema_version(conn) >= 1
        conn.close()


class TestCustomMigrationsDir:
    def test_custom_migrations_dir(self, tmp_path: object) -> None:

        tmp = Path(str(tmp_path))
        custom_dir = tmp / "custom_migrations"
        custom_dir.mkdir()
        (custom_dir / "001_test.sql").write_text("CREATE TABLE test_custom (id INTEGER PRIMARY KEY);")

        db_path = tmp / "test.db"
        conn = create_connection(db_path, migrations_dir=custom_dir)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "test_custom" in tables
        assert "player" not in tables
        conn.close()

    def test_default_migrations_dir_unchanged(self) -> None:
        conn = create_connection(":memory:")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        assert "player" in tables
        conn.close()


class TestAtomicMigrations:
    def test_failed_migration_does_not_bump_version(self, tmp_path: object) -> None:
        tmp = Path(str(tmp_path))
        mig_dir = tmp / "migrations"
        mig_dir.mkdir()
        (mig_dir / "001_good.sql").write_text("CREATE TABLE good_table (id INTEGER PRIMARY KEY);")
        (mig_dir / "002_bad.sql").write_text(
            "CREATE TABLE bad_table (id INTEGER PRIMARY KEY);\n" "INSERT INTO nonexistent_table VALUES (1);"
        )

        db_path = tmp / "test.db"
        with pytest.raises(sqlite3.OperationalError):
            create_connection(db_path, migrations_dir=mig_dir)

        # Re-open without running migrations to inspect state
        raw_conn = sqlite3.connect(str(db_path))
        assert get_schema_version(raw_conn) == 1
        tables = {row[0] for row in raw_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "good_table" in tables
        assert "bad_table" not in tables
        raw_conn.close()


class TestAttachDatabase:
    def test_attach_and_query(self, tmp_path: object) -> None:

        tmp = Path(str(tmp_path))
        main_path = tmp / "main.db"
        other_path = tmp / "other.db"

        main_conn = create_connection(main_path)
        other_conn = create_connection(other_path)

        # Insert a player in the other db
        other_conn.execute("INSERT INTO player (name_first, name_last) VALUES ('Mike', 'Trout')")
        other_conn.commit()
        other_conn.close()

        # Attach and query
        attach_database(main_conn, other_path, "other")
        rows = main_conn.execute("SELECT name_first, name_last FROM other.player").fetchall()
        assert len(rows) == 1
        assert tuple(rows[0]) == ("Mike", "Trout")
        main_conn.close()
