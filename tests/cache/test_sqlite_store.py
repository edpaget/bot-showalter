from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore

if TYPE_CHECKING:
    from pathlib import Path


class TestSqliteCacheStore:
    def test_get_returns_none_on_miss(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        assert store.get("ns", "missing") is None

    def test_put_and_get_round_trip(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        store.put("positions", "key1", '{"a": [1, 2]}', ttl_seconds=300)
        assert store.get("positions", "key1") == '{"a": [1, 2]}'

    def test_expired_entry_returns_none(self, tmp_path: Path) -> None:
        clock_time = 1000.0

        def fake_clock() -> float:
            return clock_time

        store = SqliteCacheStore(tmp_path / "cache.db", clock=fake_clock)
        store.put("ns", "k", "val", ttl_seconds=60)

        # Still valid at t=1059
        clock_time = 1059.0
        assert store.get("ns", "k") == "val"

        # Expired at t=1061
        clock_time = 1061.0
        assert store.get("ns", "k") is None

    def test_put_overwrites_existing(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        store.put("ns", "k", "old", ttl_seconds=300)
        store.put("ns", "k", "new", ttl_seconds=300)
        assert store.get("ns", "k") == "new"

    def test_invalidate_single_key(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        store.put("ns", "a", "1", ttl_seconds=300)
        store.put("ns", "b", "2", ttl_seconds=300)
        store.invalidate("ns", "a")
        assert store.get("ns", "a") is None
        assert store.get("ns", "b") == "2"

    def test_invalidate_namespace(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        store.put("ns1", "a", "1", ttl_seconds=300)
        store.put("ns1", "b", "2", ttl_seconds=300)
        store.put("ns2", "c", "3", ttl_seconds=300)
        store.invalidate("ns1")
        assert store.get("ns1", "a") is None
        assert store.get("ns1", "b") is None
        assert store.get("ns2", "c") == "3"

    def test_auto_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "deep" / "nested" / "cache.db"
        store = SqliteCacheStore(db_path)
        store.put("ns", "k", "v", ttl_seconds=60)
        assert store.get("ns", "k") == "v"

    def test_namespaces_are_isolated(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        store.put("ns1", "k", "val1", ttl_seconds=300)
        store.put("ns2", "k", "val2", ttl_seconds=300)
        assert store.get("ns1", "k") == "val1"
        assert store.get("ns2", "k") == "val2"

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        conn = store._connect()
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal_mode == "wal"

    def test_busy_timeout_set(self, tmp_path: Path) -> None:
        store = SqliteCacheStore(tmp_path / "cache.db")
        conn = store._connect()
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 5000
