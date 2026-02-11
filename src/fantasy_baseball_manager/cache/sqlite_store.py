from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from queue import Empty, Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


class SqliteConnectionPool:
    """Thread-safe connection pool for SQLite."""

    def __init__(self, db_path: Path, max_connections: int = 5) -> None:
        self._db_path = db_path
        self._max_connections = max_connections
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=max_connections)
        self._schema_lock = threading.Lock()
        self._schema_initialized = False

    def _ensure_initialized(self, conn: sqlite3.Connection) -> None:
        if self._schema_initialized:
            return
        with self._schema_lock:
            if self._schema_initialized:
                return
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  namespace TEXT NOT NULL,"
                "  key TEXT NOT NULL,"
                "  value TEXT NOT NULL,"
                "  expires_at REAL NOT NULL,"
                "  PRIMARY KEY (namespace, key)"
                ")"
            )
            self._schema_initialized = True

    def _create_connection(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=5000")
        self._ensure_initialized(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Acquire a connection from the pool, returning it when done."""
        conn: sqlite3.Connection | None = None
        try:
            conn = self._pool.get_nowait()
        except Empty:
            conn = self._create_connection()

        try:
            yield conn
        finally:
            try:
                self._pool.put_nowait(conn)
            except Exception:
                conn.close()


class SqliteCacheStore:
    def __init__(self, db_path: Path, clock: Callable[[], float] = time.time) -> None:
        self._db_path = db_path
        self._clock = clock
        self._pool = SqliteConnectionPool(db_path)

    def get(self, namespace: str, key: str) -> str | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if self._clock() >= expires_at:
                conn.execute(
                    "DELETE FROM cache WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
                conn.commit()
                return None
            return value

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        with self._pool.connection() as conn:
            expires_at = self._clock() + ttl_seconds
            conn.execute(
                "INSERT OR REPLACE INTO cache (namespace, key, value, expires_at) VALUES (?, ?, ?, ?)",
                (namespace, key, value, expires_at),
            )
            conn.commit()

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        with self._pool.connection() as conn:
            if key is not None:
                conn.execute(
                    "DELETE FROM cache WHERE namespace = ? AND key = ?",
                    (namespace, key),
                )
            else:
                conn.execute("DELETE FROM cache WHERE namespace = ?", (namespace,))
            conn.commit()
