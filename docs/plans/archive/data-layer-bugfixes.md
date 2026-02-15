# Data Layer Bug Fixes Roadmap

## Goal

Fix the nine bugs identified in the data layer (Phases 1–5) in a safe order that respects dependencies between fixes.

---

## Phase 1 — Transaction and commit discipline

Fixes Bugs 8, 9, and partially 2. These are the highest-impact issues and form the foundation for later fixes.

### Remove per-row commits from all repos

Every repo (`player_repo.py`, `batting_stats_repo.py`, `pitching_stats_repo.py`, `projection_repo.py`, `statcast_pitch_repo.py`, `load_log_repo.py`, `model_run_repo.py`) calls `self._conn.commit()` after each upsert/insert. Remove these. Transaction boundaries should be controlled by the caller, not the repo.

- Remove `self._conn.commit()` from every repo method.
- Update all repo tests to explicitly commit (or use autocommit) where needed to keep tests green.

### Add transaction management to loaders

With repos no longer auto-committing, `loader.py` (`StatsLoader.load()` and `PlayerLoader.load()`) should wrap the entire load in a single transaction and handle errors properly:

- Begin a transaction before iterating rows.
- On success, commit and write a success `LoadLog` entry.
- On any exception during upsert, rollback, write an error `LoadLog` entry (with `rows_loaded=0` and the error message), then re-raise.
- This eliminates partial-commit states and guarantees every load attempt is logged.

### Add rollback-on-release to connection pool

In `pool.py`, the `connection()` context manager should `conn.rollback()` before releasing back to the pool. This ensures no dirty transaction state leaks to the next consumer, and is safe to call even when no transaction is open.

---

## Phase 2 — Player upsert conflict handling

Fixes Bug 5. Depends on Phase 1 (transaction discipline) so that a failed upsert rolls back cleanly.

### Handle all four unique constraints in player upsert

`SqlitePlayerRepo.upsert()` currently only declares `ON CONFLICT(mlbam_id)`. The `player` table also has UNIQUE constraints on `fangraphs_id`, `bbref_id`, and `retro_id`.

Approach: use a two-step upsert pattern within a single transaction:

1. Attempt `SELECT` by `mlbam_id` (primary match key).
2. If found, `UPDATE` the existing row.
3. If not found, `INSERT` — but catch `IntegrityError` from the other unique columns. On conflict, look up the conflicting row by `fangraphs_id`/`bbref_id`/`retro_id` and update it instead, merging the IDs.

Alternatively, if the data guarantees that `mlbam_id` is always present and is the canonical key (which the Chadwick register enforces — `chadwick_row_to_player` returns `None` when `mlbam_id` is missing), document this assumption and add a guard that raises a clear error when a secondary-key conflict occurs, rather than letting SQLite raise a raw `IntegrityError`.

Test cases to add:
- Upsert a player that matches on `fangraphs_id` but not `mlbam_id`.
- Upsert a player that matches on `bbref_id` but not `mlbam_id`.
- Verify no `IntegrityError` escapes.

---

## Phase 3 — Data quality: NaN player names

Fixes Bug 7. Independent of other phases.

### Filter or clean NaN names in `chadwick_row_to_player`

In `column_maps.py`, `chadwick_row_to_player` passes `str(row["name_first"])` and `str(row["name_last"])` directly, which converts `NaN` to the literal string `"nan"`.

Fix: add NaN checks before string conversion. If both `name_first` and `name_last` are NaN, return `None` (skip the row). If only one is NaN, substitute an empty string or a placeholder like `""`.

Test cases to add:
- Row with NaN `name_first` and valid `name_last` — produces a Player with `name_first=""`.
- Row with both names NaN — returns `None`.
- Row with valid names — unchanged behavior.

---

## Phase 4 — Connection pool leak fixes

Fixes Bugs 1 and 2 (the remaining part of 2 not covered in Phase 1). Independent of Phases 2–3.

### Close connections on release when pool is closed

In `pool.py`, `release()` currently silently discards connections when `self._closed` is True. Fix: call `conn.close()` in that branch.

### Close checked-out connections on pool shutdown

`close_all()` can only drain connections currently in the queue. Connections checked out by other threads are lost. Fix: track all created connections in a list at `__init__` time. In `close_all()`, iterate the list and close every connection, not just those in the queue.

Test cases to add:
- Close pool while a connection is checked out — verify the connection is closed after release.
- Release a connection after pool is closed — verify it is closed, not leaked.

---

## Phase 5 — Migration robustness and cleanup

Fixes Bugs 3, 4, and 6. Lower severity; can be done last.

### Make migrations idempotent

In `connection.py`, `_run_migrations` uses `executescript()` which auto-commits each statement. If a multi-statement migration partially applies and the `schema_version` insert never runs, retrying re-executes already-applied statements.

Fix: replace `executescript()` with manual statement splitting and execution inside an explicit transaction. Wrap each migration file's execution in `BEGIN` / `COMMIT` with the `schema_version` insert inside the same transaction, so either the entire migration applies or none of it does.

Migration 002 specifically needs attention: `ALTER TABLE projection ADD COLUMN source_type` is not idempotent. Either:
- Check `PRAGMA table_info(projection)` before altering, or
- Wrap in a try/except for the "duplicate column" error.

### Remove dead `SCHEMA_VERSION` constant

Delete `db/schema.py` (which contains only `SCHEMA_VERSION = 2`) and remove any imports. The migration runner already tracks versions via the `schema_version` table.

### Switch to `sqlite3.Row` row factory

Set `conn.row_factory = sqlite3.Row` in `create_connection()`. Then update all `_row_to_*` methods to use column-name access (`row["id"]`, `row["name_first"]`) instead of positional indexing (`row[0]`, `row[1]`). This prevents silent data corruption if columns are ever reordered or added.

This is a broad change touching every repo, so:
- Set the row factory in one place (`connection.py`).
- Update each repo's `_row_to_*` method one at a time, with tests passing after each.

---

## Phase order and dependencies

```
Phase 1 (transactions)
  ├── Phase 2 (player upsert) — depends on Phase 1
  └── Phase 4 (pool leaks) — partially depends on Phase 1
Phase 3 (NaN names) — independent
Phase 5 (migrations, row factory) — independent, do last due to breadth
```

Phases 1, 3, and 5 can be started in parallel. Phase 2 should follow Phase 1. Phase 4's rollback-on-release piece is in Phase 1; the remaining leak fixes are independent.
