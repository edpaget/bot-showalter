import sqlite3
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def create_connection(
    path: str | Path,
    *,
    check_same_thread: bool = True,
    migrations_dir: Path | None = None,
) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode, foreign keys, and pending migrations applied."""
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _run_migrations(conn, migrations_dir=migrations_dir)
    return conn


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version, or 0 if no migrations have run."""
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    except sqlite3.OperationalError:
        return 0
    return row[0] if row and row[0] is not None else 0


def attach_database(conn: sqlite3.Connection, path: str | Path, name: str) -> None:
    """Attach another SQLite database file to an existing connection."""
    conn.execute(f"ATTACH DATABASE ? AS [{name}]", (str(path),))


def _run_migrations(conn: sqlite3.Connection, *, migrations_dir: Path | None = None) -> None:
    """Apply any pending numbered .sql migration files."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version ("
        "    version INTEGER PRIMARY KEY,"
        "    applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )

    current_version = get_schema_version(conn)

    effective_dir = migrations_dir if migrations_dir is not None else _MIGRATIONS_DIR
    migration_files = sorted(effective_dir.glob("*.sql"))
    for migration_file in migration_files:
        version = int(migration_file.stem.split("_")[0])
        if version <= current_version:
            continue
        sql = migration_file.read_text()
        conn.executescript(sql)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        conn.commit()

    # Re-enable foreign keys after executescript (which implicitly commits and may reset pragmas)
    conn.execute("PRAGMA foreign_keys=ON")
