# Unified Draft Engine Roadmap

The draft system currently has two orchestration layers — `DraftSession` (CLI REPL) and `SessionManager` (web/GraphQL) — that both wrap `DraftEngine` but duplicate session initialization, keeper handling, pick persistence, position auto-detection, and Yahoo integration. This roadmap extracts a single `DraftSessionService` that owns all draft orchestration, making sessions interchangeable between CLI and web. The end state: a draft started in the web UI can be queried and acted on by CLI commands (e.g., a Claude agent running `fbm draft recommend --session <id>`).

The core `DraftEngine` (state machine) and pure business-logic functions (`recommend()`, `build_draft_board()`, `analyze_roster()`, etc.) are already well-factored and shared. The problem is entirely in the orchestration/session layer.

## Status

| Phase | Status |
|-------|--------|
| 1 — Extract DraftSessionService | not started |
| 2 — Migrate web to DraftSessionService | not started |
| 3 — Migrate CLI to DraftSessionService | not started |
| 4 — Cross-client session commands | not started |
| 5 — Event bus for CLI consumers | not started |

## Phase 1: Extract DraftSessionService

Extract the shared orchestration logic from `SessionManager` and `DraftSession` into a new `DraftSessionService` in the services layer.

### Context

Today `SessionManager` (web layer, `web/session_manager.py`) handles session creation, keeper snapshots, pick persistence, trade management, and engine caching. `DraftSession` (services layer, `services/draft_session.py`) independently handles pick persistence, session creation, Yahoo pick draining, and recommendation display. Both duplicate keeper filtering, position auto-detection, and session lifecycle management. The new service consolidates these into one place that neither presentation layer owns.

### Steps

1. Create `services/draft_session_service.py` with a `DraftSessionService` class. Its constructor accepts the same repo/service dependencies that `SessionManager` currently takes (session repo, valuation repo, player repo, ADP repo, profile service, league settings, projection repo, keeper repos, etc.).
2. Move `SessionManager.start_session()` logic into `DraftSessionService.start_session()` — keeper snapshot building, valuation filtering, board construction, engine creation, DB persistence. Return `(session_id, DraftEngine)`.
3. Move `SessionManager.get_engine()` (lazy load from DB with cache) into `DraftSessionService.get_engine()`.
4. Move `SessionManager.pick()`, `undo()`, `trade_picks()`, `undo_trade()`, `evaluate_trade()` into `DraftSessionService` — each mutates the engine and persists to DB.
5. Move `SessionManager.end_session()`, `list_sessions()`, `get_team_names()`, `get_keepers()` into the service.
6. Move `SessionManager`'s category balance / weak categories tracking into the service (`get_category_balance_fn()`, `get_weak_categories()`).
7. Extract `auto_detect_position()` (currently duplicated in `DraftSession._handle_pick()` and the web schema `pick` mutation) into a method on `DraftSessionService` or a standalone function in the services layer.
8. Move `SessionManager._build_player_pool()` and `_build_keeper_snapshot()` into the service as private methods.
9. Add `recommend()` and `board()` convenience methods that delegate to the existing pure functions with the session's cached state (projections, league, category balance fn).
10. Add `persist_external_pick()` for Yahoo poller integration.
11. Define protocols for any web-layer dependencies that shouldn't leak into services (e.g., `ValuationAdjuster`, `KeeperCostDeriver` are already protocols — keep them).
12. Write unit tests for `DraftSessionService` covering: start with/without keepers, pick/undo lifecycle, trade/undo-trade, session resume from DB, auto-position detection, list/end sessions.

### Acceptance criteria

- `DraftSessionService` exists in `services/` with the full public API: `start_session`, `get_engine`, `pick`, `undo`, `trade_picks`, `undo_trade`, `evaluate_trade`, `end_session`, `list_sessions`, `recommend`, `available`, `roster`, `needs`, `balance`, `category_needs`, `get_keepers`, `get_team_names`, `auto_detect_position`, `persist_external_pick`.
- All methods are covered by unit tests using injected fakes (no DB or web dependencies).
- `SessionManager` still exists and works (not yet migrated) — this phase only extracts, it doesn't remove the old code.
- Architecture tests pass (no forbidden layer imports).

## Phase 2: Migrate web to DraftSessionService

Replace `SessionManager` usage in GraphQL resolvers with `DraftSessionService`.

### Context

After Phase 1, `DraftSessionService` has the full API. The web layer's `SessionManager` is now redundant. GraphQL resolvers currently call `SessionManager` methods and do some inline logic (auto-position detection, arbitrage computation, building pick results). This phase makes resolvers thin wrappers over `DraftSessionService`.

### Steps

1. Update `AppContext` (or however the web layer stores shared state) to hold a `DraftSessionService` instance instead of (or alongside) `SessionManager`.
2. Update each GraphQL mutation to delegate to `DraftSessionService`:
   - `start_session` → `service.start_session()`
   - `pick` → `service.auto_detect_position()` + `service.pick()`
   - `undo` → `service.undo()`
   - `trade_picks` / `undo_trade` → `service.trade_picks()` / `service.undo_trade()`
   - `end_session` → `service.end_session()`
3. Update each GraphQL query to delegate to `DraftSessionService`:
   - `recommendations` → `service.recommend()`
   - `roster`, `needs`, `balance`, `category_needs` → service methods
   - `available` → `service.available()`
   - `keepers` → `service.get_keepers()`
   - `sessions` → `service.list_sessions()`
   - `evaluate_trade` → `service.evaluate_trade()`
4. Keep event-bus publishing in the resolvers (presentation concern — the service doesn't know about GraphQL subscriptions).
5. Keep Yahoo poll start/stop in the web layer (it manages the async poller lifecycle), but have the poller call `service.persist_external_pick()`.
6. Delete `SessionManager` class and `web/session_manager.py`.
7. Verify all existing frontend integration tests and GraphQL tests pass.

### Acceptance criteria

- `web/session_manager.py` is deleted.
- All GraphQL mutations and queries delegate to `DraftSessionService`.
- Event publishing remains in resolvers (not in the service).
- All web/frontend tests pass unchanged (behavior is identical from the client's perspective).
- Yahoo poll manager calls `DraftSessionService.persist_external_pick()`.

## Phase 3: Migrate CLI to DraftSessionService

Replace `DraftSession`'s inline persistence and session management with `DraftSessionService`, making CLI sessions DB-compatible with web sessions.

### Context

`DraftSession` currently manages its own DB persistence (`_persist_pick`, `_persist_undo`, `_maybe_create_session`) and session creation. After this phase, `DraftSession` becomes a thin REPL wrapper: it parses commands, calls `DraftSessionService` methods, and formats output. Sessions created by CLI are indistinguishable from web sessions in the DB.

### Steps

1. Change `DraftSession` constructor to accept a `DraftSessionService` and `session_id` instead of raw `DraftEngine`, `session_repo`, and board-building dependencies.
2. Replace `_maybe_create_session()` — session creation now happens before constructing `DraftSession`, via `service.start_session()`.
3. Replace `_handle_pick()` to call `service.auto_detect_position()` + `service.pick()` instead of `engine.pick()` + `_persist_pick()`.
4. Replace `_handle_undo()` to call `service.undo()` instead of `engine.undo()` + `_persist_undo()`.
5. Replace `best` command to call `service.recommend()`.
6. Replace `balance`, `needs`, `roster`, `pool` commands to call service methods.
7. Replace trade commands to call `service.trade_picks()` / `service.undo_trade()`.
8. Update `draft_start` CLI command (`cli/commands/draft.py`) to:
   - Build `DraftSessionService` (via factory) with all repos
   - Call `service.start_session()` or `service.get_engine()` for resume
   - Pass service + session_id to `DraftSession`
9. Update `draft_report`, `draft_sessions`, `draft_delete` CLI commands to use `DraftSessionService`.
10. Remove `_persist_pick`, `_persist_undo`, `_maybe_create_session` from `DraftSession`.
11. Remove `save_draft` / `load_draft` JSON persistence (DB is now the canonical store; JSON export can remain as a separate export command if needed).
12. Update `DraftSession` tests to inject a fake `DraftSessionService`.
13. Wire `DraftSessionService` construction in `cli/factory.py`.

### Acceptance criteria

- `DraftSession` no longer directly uses `DraftSessionRepo` or `DraftEngine` — it delegates all state mutations to `DraftSessionService`.
- CLI `draft start` creates sessions via `DraftSessionService.start_session()`.
- CLI `draft start --session-id <id>` resumes a session (including one started from the web UI).
- All CLI draft tests pass.
- `save_draft` / `load_draft` JSON functions are removed (or moved to an export-only path).

## Phase 4: Cross-client session commands

Add CLI commands that operate on any active session (including web-started ones), enabling Claude agent interaction with live drafts.

### Context

After Phase 3, CLI and web share `DraftSessionService` and the same DB schema. But the CLI still only interacts with sessions through the REPL (`draft start`). This phase adds non-interactive CLI commands that a Claude agent (or script) can invoke against any session by ID.

### Steps

1. Add `draft recommend --session <id> [--position POS] [--limit N]` — prints top recommendations for the session's current state.
2. Add `draft pick --session <id> <player> [--position POS] [--price N]` — executes a pick in an existing session (non-interactive).
3. Add `draft undo --session <id>` — undoes the last pick.
4. Add `draft roster --session <id> [--team N]` — shows a team's roster.
5. Add `draft needs --session <id>` — shows unfilled slots.
6. Add `draft balance --session <id>` — shows category balance.
7. Add `draft available --session <id> [--position POS] [--limit N]` — shows available players.
8. Add `draft falls --session <id>` — shows ADP fallers (arbitrage opportunities).
9. Add `draft trade-eval --session <id> --gives ... --receives ...` — evaluates a pick trade.
10. Add `draft status --session <id>` — shows current pick number, team on clock, budget remaining.
11. All commands construct `DraftSessionService` via factory, call the appropriate method, and print formatted output.
12. Document the CLI commands in the fbm skill definition so Claude agents can discover and use them.

### Acceptance criteria

- Each command works against any session regardless of whether it was started from CLI or web.
- `draft recommend --session <id>` returns the same recommendations as the web UI's recommendation panel for the same session state.
- A Claude agent can run `fbm draft recommend --session <id>` and get actionable output.
- Commands are non-interactive (no REPL, no prompts) — suitable for programmatic use.
- All new commands have tests.

## Phase 5: Event bus for CLI consumers

Add optional event subscription so CLI tools can react to picks made in the web UI in real time.

### Context

The web UI publishes draft events (picks, undos, trades, arbitrage alerts) via an async event bus to GraphQL subscriptions. CLI commands from Phase 4 are request/response — they show state at a point in time but don't stream updates. This phase adds a lightweight event listener so a CLI session or agent can watch for new picks and react.

### Steps

1. Extract the event bus interface into a protocol in the services layer (currently it's web-only in `web/event_bus.py` or similar).
2. Add a `draft watch --session <id>` CLI command that subscribes to session events and prints them as they arrive (picks, trades, arbitrage alerts).
3. Add `--auto-recommend` flag to `draft watch` that prints updated recommendations after each pick event.
4. Ensure the event bus works across processes — if the web server publishes a pick, a CLI watcher in a separate process receives it. This likely requires a lightweight IPC mechanism (e.g., SQLite WAL polling, Unix socket, or Redis pub/sub depending on deployment).
5. Update `DraftSessionService` to optionally accept and publish to an event bus when picks/trades occur, so both web resolvers and CLI can trigger events.
6. Update the REPL (`DraftSession`) to optionally subscribe to events, enabling it to show picks made in the web UI in real time.

### Acceptance criteria

- `draft watch --session <id>` prints picks/trades as they happen in the web UI.
- `draft watch --session <id> --auto-recommend` prints updated recommendations after each pick.
- Events are delivered cross-process (web server → CLI watcher).
- Existing web subscriptions continue to work unchanged.
- REPL session can show picks made via web UI (when connected to same session).

## Ordering

**Phase 1 → Phase 2 → Phase 3** must be sequential — each phase depends on the prior extraction/migration.

**Phase 4** depends on Phase 3 (CLI must use `DraftSessionService` before non-interactive commands can be added). This is the critical phase for enabling Claude agent interaction.

**Phase 5** is independent of Phase 4 but benefits from it. It can be deferred or skipped if polling-based CLI commands (Phase 4) are sufficient for the agent use case.

**Suggested priority:** Phases 1–4 are the core value. Phase 5 is a nice-to-have for real-time reactivity but adds infrastructure complexity (cross-process eventing). Start with Phases 1–3 to unify the engine, then Phase 4 to unlock the agent workflow.
