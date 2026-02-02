from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class SqliteCacheStore:
    def __init__(self, db_path: Path, clock: Callable[[], float] = time.time) -> None:
        self._db_path = db_path
        self._clock = clock
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  namespace TEXT NOT NULL,"
                "  key TEXT NOT NULL,"
                "  value TEXT NOT NULL,"
                "  expires_at REAL NOT NULL,"
                "  PRIMARY KEY (namespace, key)"
                ")"
            )
        return self._conn

    def get(self, namespace: str, key: str) -> str | None:
        conn = self._connect()
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
        conn = self._connect()
        expires_at = self._clock() + ttl_seconds
        conn.execute(
            "INSERT OR REPLACE INTO cache (namespace, key, value, expires_at) VALUES (?, ?, ?, ?)",
            (namespace, key, value, expires_at),
        )
        conn.commit()

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        conn = self._connect()
        if key is not None:
            conn.execute(
                "DELETE FROM cache WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
        else:
            conn.execute("DELETE FROM cache WHERE namespace = ?", (namespace,))
        conn.commit()
