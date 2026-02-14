import sqlite3
import threading

import pytest

from fantasy_baseball_manager.db.connection import get_schema_version
from fantasy_baseball_manager.db.pool import ConnectionPool
from fantasy_baseball_manager.db.schema import SCHEMA_VERSION


class TestConnectionPool:
    def test_get_returns_connection(self) -> None:
        pool = ConnectionPool(":memory:", size=2)
        conn = pool.get()
        assert isinstance(conn, sqlite3.Connection)
        pool.release(conn)
        pool.close_all()

    def test_get_and_release_cycle(self) -> None:
        pool = ConnectionPool(":memory:", size=1)
        conn1 = pool.get()
        pool.release(conn1)
        conn2 = pool.get()
        assert conn2 is conn1
        pool.release(conn2)
        pool.close_all()

    def test_exhaustion_raises_timeout_error(self) -> None:
        pool = ConnectionPool(":memory:", size=1)
        conn = pool.get()
        with pytest.raises(TimeoutError):
            pool.get(timeout=0.01)
        pool.release(conn)
        pool.close_all()

    def test_context_manager(self) -> None:
        pool = ConnectionPool(":memory:", size=2)
        with pool.connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
        pool.close_all()

    def test_context_manager_releases_on_exception(self) -> None:
        pool = ConnectionPool(":memory:", size=1)
        with pytest.raises(ValueError, match="test error"):
            with pool.connection() as _conn:
                raise ValueError("test error")
        # Connection should be released back, so we can get it again
        conn = pool.get(timeout=0.1)
        assert isinstance(conn, sqlite3.Connection)
        pool.release(conn)
        pool.close_all()

    def test_close_all_prevents_further_use(self) -> None:
        pool = ConnectionPool(":memory:", size=2)
        pool.close_all()
        with pytest.raises(RuntimeError):
            pool.get()

    def test_thread_safety(self) -> None:
        pool = ConnectionPool(":memory:", size=3)
        results: list[bool] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                with pool.connection() as conn:
                    conn.execute("SELECT 1")
                    results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 6

    def test_file_based_migrations_run_once(self, tmp_path: object) -> None:
        from pathlib import Path

        db_path = Path(str(tmp_path)) / "pool_test.db"
        pool = ConnectionPool(db_path, size=2)
        with pool.connection() as conn:
            assert get_schema_version(conn) == SCHEMA_VERSION
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            assert "player" in tables
        pool.close_all()
