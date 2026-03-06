import sqlite3
import threading

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.db.pool import (
    ConnectionPool,
    SingleConnectionProvider,
)
from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class TestConnectionProviderProtocol:
    def test_is_runtime_checkable(self) -> None:
        assert isinstance(SingleConnectionProvider(sqlite3.connect(":memory:")), ConnectionProvider)

    def test_connection_pool_satisfies_protocol(self) -> None:
        pool = ConnectionPool(":memory:", size=1)
        assert isinstance(pool, ConnectionProvider)
        pool.close_all()


class TestSingleConnectionProvider:
    def test_yields_same_connection(self) -> None:
        raw = sqlite3.connect(":memory:")
        provider = SingleConnectionProvider(raw)
        with provider.connection() as conn:
            assert conn is raw

    def test_yields_same_connection_on_repeated_calls(self) -> None:
        raw = sqlite3.connect(":memory:")
        provider = SingleConnectionProvider(raw)
        with provider.connection() as c1:
            pass
        with provider.connection() as c2:
            assert c1 is c2

    def test_no_rollback_on_exit(self) -> None:
        raw = sqlite3.connect(":memory:")
        raw.execute("CREATE TABLE t (x INTEGER)")
        provider = SingleConnectionProvider(raw)
        with provider.connection() as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            # Do NOT commit inside the context manager
        # After exiting, the uncommitted row should still be visible
        # (no rollback happened)
        row = raw.execute("SELECT x FROM t").fetchone()
        assert row == (1,)

    def test_no_rollback_on_exception(self) -> None:
        raw = sqlite3.connect(":memory:")
        raw.execute("CREATE TABLE t (x INTEGER)")
        provider = SingleConnectionProvider(raw)
        try:
            with provider.connection() as conn:
                conn.execute("INSERT INTO t VALUES (42)")
                raise ValueError("boom")
        except ValueError:
            pass
        # Row should still be visible — no rollback
        row = raw.execute("SELECT x FROM t").fetchone()
        assert row == (42,)


class TestConnectionPoolMultiThreaded:
    """Integration test: concurrent reads through a ConnectionPool don't raise."""

    def test_concurrent_reads_through_repos(self, tmp_path: object) -> None:
        db_path = f"{tmp_path}/test.db"
        # Seed data via a temporary connection
        setup_conn = sqlite3.connect(db_path)
        setup_conn.execute("CREATE TABLE IF NOT EXISTS players (id INTEGER PRIMARY KEY, name TEXT)")
        setup_conn.execute("INSERT INTO players VALUES (1, 'Test Player')")
        setup_conn.commit()
        setup_conn.close()

        pool = ConnectionPool(db_path, size=4)
        container = AnalysisContainer(pool)
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def _read_from_thread() -> None:
            try:
                barrier.wait(timeout=5)
                # Access a repo — this checks out a connection from the pool
                container.player_repo.all()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_read_from_thread) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        pool.close_all()
        assert errors == [], f"Threads raised: {errors}"
