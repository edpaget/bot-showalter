import sqlite3

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
