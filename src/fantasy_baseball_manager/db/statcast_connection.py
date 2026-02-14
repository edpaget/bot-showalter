import sqlite3
from pathlib import Path

from fantasy_baseball_manager.db.connection import create_connection

_STATCAST_MIGRATIONS_DIR = Path(__file__).parent / "statcast_migrations"


def create_statcast_connection(path: str | Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection for the statcast database with its own migrations."""
    return create_connection(path, check_same_thread=check_same_thread, migrations_dir=_STATCAST_MIGRATIONS_DIR)
