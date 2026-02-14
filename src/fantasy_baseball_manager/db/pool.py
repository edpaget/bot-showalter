import queue
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from fantasy_baseball_manager.db.connection import create_connection


class ConnectionPool:
    """Thread-safe pool of SQLite connections backed by a queue."""

    def __init__(self, path: str | Path, *, size: int = 5) -> None:
        self._closed = False
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=size)
        for _ in range(size):
            conn = create_connection(path, check_same_thread=False)
            self._pool.put(conn)

    def get(self, *, timeout: float | None = None) -> sqlite3.Connection:
        """Check out a connection from the pool.

        Raises RuntimeError if pool is closed.
        Raises TimeoutError if no connection is available within timeout.
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("No connection available in pool")

    def release(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        if not self._closed:
            self._pool.put(conn)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection]:
        """Context manager that checks out and auto-releases a connection."""
        conn = self.get()
        try:
            yield conn
        finally:
            self.release(conn)

    def close_all(self) -> None:
        """Drain the pool and close all connections."""
        self._closed = True
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
