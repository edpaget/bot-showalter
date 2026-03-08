# Web API Hardening Roadmap

Fix bugs, encapsulation violations, and design issues identified in the web module code review (phases 1-3 of web-ui-foundation). These are all backend-only changes that improve correctness and maintainability before the React frontend work begins.

## Status

| Phase | Status |
|-------|--------|
| 1 — Subscription leak and error handling | done (2026-03-08) |
| 2 — SessionManager encapsulation | not started |
| 3 — Session system/version persistence | not started |

## Phase 1: Subscription leak and error handling

Fix the subscription queue leak and replace the repeated `assert mgr is not None` pattern with a proper helper.

### Context

The `draft_events` subscription only unsubscribes on `CancelledError`. If any other exception propagates from the `while True` loop, the queue stays registered in the `EventBus` forever, leaking memory and delivering events to a dead consumer. Separately, 11 resolvers use `assert mgr is not None` which produces unhelpful `AssertionError` stack traces instead of meaningful GraphQL errors.

### Steps

1. Change the `draft_events` subscription in `schema.py` to use `try/finally` instead of `except CancelledError` for unsubscribe cleanup.
2. Write a test that verifies unsubscribe is called even when a non-cancellation exception occurs in the subscription loop.
3. Extract a `_get_session_manager(info) -> SessionManager` helper in `schema.py` that raises a descriptive `ValueError("Session management is not configured")` when `session_manager` is `None`. Replace all 11 `assert mgr is not None` sites with calls to this helper.
4. Remove the unused `_player_repo` field from the `YahooPollerManager` dataclass and update any constructor call sites (including tests).

### Acceptance criteria

- Subscription queue is unsubscribed on any exception, not just `CancelledError`.
- Test proves unsubscribe happens on arbitrary exceptions.
- No `assert mgr is not None` remains in `schema.py`.
- All resolvers that need `SessionManager` use the new helper and produce a clear error message when it's `None`.
- `YahooPollerManager` has no unused fields.
- All existing tests pass.

## Phase 2: SessionManager encapsulation

Add proper public methods to `SessionManager` so resolvers don't reach into private attributes, and clean up test infrastructure.

### Context

The `sessions()` resolver accesses `mgr._repo.list_sessions()` and `mgr._repo.count_picks()` directly — a violation of encapsulation that couples the GraphQL layer to the repo's internal API. If the repo interface changes, schema.py breaks even though SessionManager's public API might be stable. Similarly, the test conftest accesses `provider._conn.commit()` to flush test data.

### Steps

1. Add a `list_sessions(*, league: str | None, season: int | None, status: str | None) -> list[DraftSessionSummaryType]` method (or a simpler return type like a list of tuples/dataclasses) to `SessionManager` that wraps the repo's `list_sessions()` and `count_picks()` calls.
2. Update the `sessions()` resolver in `schema.py` to call the new public method instead of accessing `mgr._repo` directly.
3. Write a test that exercises the new `list_sessions` method on `SessionManager` directly.
4. In `tests/web/conftest.py`, replace `provider._conn.commit()` with the public `provider.connection()` context manager to commit test data, avoiding private attribute access.

### Acceptance criteria

- `schema.py` does not access any private (`_`-prefixed) attributes of `SessionManager`.
- `SessionManager.list_sessions()` encapsulates the repo interaction.
- Test conftest uses only public APIs of `SingleConnectionProvider`.
- All existing tests pass.

## Phase 3: Session system/version persistence

Store the valuation `system` and `version` in the `draft_session` table so session hydration uses the correct player pool.

### Context

`SessionManager.get_engine()` hardcodes `"zar"` and `"1.0"` when rehydrating a session from the database because `DraftSessionRecord` and the `draft_session` table don't store these values. If a future valuation system is added, rehydrated sessions would silently use the wrong player pool. This requires a schema migration, a domain model change, and updates to the repo, session manager, and GraphQL layer.

### Steps

1. Add a migration (`027_draft_session_system_version.sql` or next available number) that adds `system TEXT NOT NULL DEFAULT 'zar'` and `version TEXT NOT NULL DEFAULT '1.0'` columns to the `draft_session` table. The defaults ensure existing rows are valid.
2. Add `system` and `version` fields to `DraftSessionRecord` (with defaults matching the migration).
3. Update `SqliteDraftSessionRepo.create_session()` and `load_session()` to read/write the new columns.
4. Update `SessionManager.start_session()` to store `system` and `version` in the record.
5. Update `SessionManager.get_engine()` to use `record.system` and `record.version` instead of hardcoded values.
6. Update `DraftSessionSummaryType` to include `system` and `version` fields.
7. Write tests: create a session with non-default system/version, rehydrate it, and verify the correct player pool is used.

### Acceptance criteria

- `draft_session` table has `system` and `version` columns with backward-compatible defaults.
- `DraftSessionRecord` includes `system` and `version`.
- `get_engine()` uses the session's stored system/version, not hardcoded values.
- A session created with a non-default system/version rehydrates correctly.
- All existing tests pass without modification (defaults cover them).

## Ordering

All three phases are independent and can be implemented in any order. The suggested priority is:

1. **Phase 1** first — fixes a resource leak (bug) and improves error messages. Smallest, safest change.
2. **Phase 2** second — improves encapsulation. No schema changes, no migration risk.
3. **Phase 3** third — requires a database migration and touches more layers. Lowest urgency since only one valuation system exists today.
