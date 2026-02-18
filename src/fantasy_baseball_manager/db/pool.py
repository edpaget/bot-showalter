import logging
import queue
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from fantasy_baseball_manager.db.connection import create_connection

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Thread-safe pool of SQLite connections backed by a queue."""

    def __init__(self, path: str | Path, *, size: int = 5) -> None:
        logger.debug("Creating connection pool: size=%d", size)
        self._closed = False
        self._all_conns: list[sqlite3.Connection] = []
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=size)
        for _ in range(size):
            conn = create_connection(path, check_same_thread=False)
            self._all_conns.append(conn)
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
            logger.warning("Connection pool exhausted")
            raise TimeoutError("No connection available in pool")

    def release(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        if not self._closed:
            self._pool.put(conn)
        else:
            conn.close()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection]:
        """Context manager that checks out and auto-releases a connection."""
        conn = self.get()
        try:
            yield conn
        finally:
            conn.rollback()
            self.release(conn)

    def close_all(self) -> None:
        """Close all connections, including checked-out ones."""
        logger.debug("Closing %d connections", len(self._all_conns))
        self._closed = True
        for conn in self._all_conns:
            conn.close()
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
