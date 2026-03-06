# Yahoo Live Draft Fixes

Fix three bugs discovered when testing `fbm yahoo draft-live` against a completed 2025 Yahoo draft: snake-order enforcement rejects out-of-order picks, position case mismatch between Yahoo (uppercase) and league config (lowercase), and SQLite thread-safety violation in the background draft poller.

## Status

| Phase | Status |
|-------|--------|
| 1 — Position case normalization | done (2026-03-06) |
| 2 — Live draft team ordering | done (2026-03-06) |
| 3 — Thread-safe draft poller | not started |

## Phase 1: Position case normalization

Normalize positions to uppercase throughout the draft system so Yahoo positions (`SS`, `SP`, `OF`) match roster slot keys.

### Context

Yahoo API returns positions in uppercase (`SS`, `2B`, `SP`, `RP`, `OF`). League configs in `fbm.toml` define positions in lowercase (`ss`, `2b`, `sp`). `build_draft_roster_slots()` passes positions through from `league.positions` without normalizing, but hardcodes `UTIL` and `P` in uppercase — so slots are already a mix. When `DraftEngine.pick()` checks `if position not in config.roster_slots`, Yahoo uppercase positions fail.

### Steps

1. Normalize `build_draft_roster_slots()` to uppercase all position keys from `league.positions`.
2. Normalize the `position` argument in `DraftEngine.pick()` to uppercase before validation.
3. Normalize `yahoo_pick.position` to uppercase in `ingest_yahoo_pick()` before passing to `pick_fn`.
4. Update existing tests that use lowercase positions in roster slot configs to use uppercase.
5. Verify `auto_detect_position()` in `draft_session.py` works with uppercase slots.

### Acceptance criteria

- `build_draft_roster_slots()` returns all-uppercase keys.
- `DraftEngine.pick()` accepts positions in any case.
- `ingest_yahoo_pick()` passes uppercase positions to the engine.
- All existing draft tests pass after normalization.

## Phase 2: Live draft team ordering

Allow Yahoo draft ingestion to record picks for any team at any pick number, instead of enforcing strict snake order.

### Context

`DraftEngine.pick()` enforces snake team order: pick 1 must be team 1, pick 2 must be team 2, etc. Yahoo live drafts record picks in the order they happened on Yahoo, where any team can pick at any time (keeper picks, traded picks, commissioner overrides). When replaying a completed Yahoo draft, nearly every pick fails with "Wrong team: expected team X, got team Y".

The snake validation is correct for the interactive REPL (where the user manually records picks in order), but wrong for Yahoo ingestion where picks arrive in Yahoo's recorded order with explicit team assignments.

### Steps

1. Add `DraftFormat.LIVE` to the `DraftFormat` enum — a format where picks can come from any team in any order.
2. In `DraftEngine.pick()`, skip the snake team validation when `format == DraftFormat.LIVE`.
3. In `yahoo_draft_setup.py`, detect Yahoo's `draft_type` and use `DraftFormat.LIVE` for live drafts instead of always choosing between `SNAKE` and `AUCTION`.
4. Update `DraftSession._handle_pick()` to handle `LIVE` format — use the user's team (like auction) instead of `team_on_clock()`.
5. Update `_show_status()` to handle `LIVE` format display.
6. Write tests: pick any team at any pick number in LIVE mode, reject invalid teams, verify pool/roster tracking still works.

### Acceptance criteria

- Yahoo live draft picks are accepted regardless of team order.
- Pool and roster tracking remain correct (players removed, rosters populated).
- Interactive session still works in LIVE mode (picks default to user's team).
- Snake and auction formats are unaffected.

## Phase 3: Thread-safe draft poller

Fix the SQLite `ProgrammingError` when the background draft poller thread accesses the database.

### Context

`YahooDraftPoller` runs `fetch_draft_results()` in a daemon thread. The `YahooDraftSource` it uses calls `YahooPlayerMapper.resolve()`, which queries the database through `yahoo_player_map_repo` and `player_repo`. These repos share the main thread's `SingleConnectionProvider(conn)`, but SQLite's default `check_same_thread=True` rejects cross-thread usage.

A `ConnectionPool` class already exists in `src/fantasy_baseball_manager/db/pool.py` with `check_same_thread=False` connections, but the draft poller setup doesn't use it.

### Steps

1. In `yahoo_draft_live` command, create a separate `ConnectionPool` (size 1) for the poller's repos instead of sharing the main thread's connection.
2. Build a separate `YahooPlayerMapper` backed by the pool's connection provider for the `YahooDraftSource` passed to the poller.
3. Ensure the pool is closed in the `finally` block alongside `poller.stop()`.
4. Write a test that verifies `YahooDraftPoller.poll_once()` works when backed by a `ConnectionPool` provider (simulate cross-thread access).

### Acceptance criteria

- Background poller fetches draft results without `ProgrammingError`.
- Player mapping in the poller thread works correctly.
- Pool connection is cleaned up on session exit.
- Main thread's connection is unaffected.

## Ordering

Phase 1 (positions) is independent and simplest — fix first. Phase 2 (team ordering) is independent of phase 1 but higher impact. Phase 3 (thread safety) is independent but only matters for live polling (not replay). All three can technically be done in parallel, but the recommended order is 1 -> 2 -> 3 for incremental validation.
