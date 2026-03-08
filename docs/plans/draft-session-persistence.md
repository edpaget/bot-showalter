# Draft Session Persistence Roadmap

Draft sessions currently support manual JSON save/load via the `save` command and `--resume` flag, but picks are only persisted when the user explicitly saves or gracefully quits. If the process crashes, is killed, or the terminal is closed, unsaved picks are lost. This roadmap moves draft session persistence into the SQLite database with write-ahead saves on every mutation, enabling crash-safe resume and future analytics over draft history.

## Status

| Phase | Status |
|-------|--------|
| 1 — Schema and repo layer | done (2026-03-07) |
| 2 — Auto-persist on every mutation | done (2026-03-07) |
| 3 — Session discovery and resume UX | in progress |

## Phase 1: Schema and repo layer

Create SQLite tables for draft sessions and picks, plus a repository that can save and load them.

### Context

The project uses `ConnectionProvider`-backed `Sqlite*Repo` classes throughout (e.g., `SqliteYahooDraftRepo`, `SqliteExperimentRepo`). Draft state is currently modeled as `DraftConfig` (frozen dataclass) + a list of `DraftPick` (frozen dataclass). Storing these relationally makes them queryable for analytics — e.g., "how often did I draft a catcher in the first 5 rounds across all drafts?" or "what was my average roster value by season?".

### Steps

1. Design the schema:
   - `draft_session` table: `id INTEGER PRIMARY KEY`, `league TEXT`, `season INT`, `teams INT`, `format TEXT`, `user_team INT`, `roster_slots TEXT` (JSON), `budget INT`, `status TEXT` (`in_progress` / `complete`), `created_at TEXT`, `updated_at TEXT`. Unique on `(league, season, created_at)` to allow multiple sessions per league/season.
   - `draft_pick` table: `id INTEGER PRIMARY KEY`, `session_id INT REFERENCES draft_session(id)`, `pick_number INT`, `team INT`, `player_id INT`, `player_name TEXT`, `position TEXT`, `price INT`. Unique on `(session_id, pick_number)`.
2. Add a `DraftSessionRow` domain dataclass (or extend existing types) to represent a persisted session with its id and metadata.
3. Create `SqliteDraftSessionRepo` with methods: `create_session(config, league) -> int`, `save_pick(session_id, pick)`, `delete_pick(session_id, pick_number)` (for undo), `load_session(session_id) -> (config, picks)`, `list_sessions(league?, season?) -> list`, `update_status(session_id, status)`, `update_timestamp(session_id)`.
4. Add the `load_draft_from_db(session_id, players, repo) -> DraftEngine` function that replays picks (parallel to existing `load_draft` for JSON).
5. Add the migration to `schema.sql` (or wherever DDL lives).
6. Write tests for the repo: round-trip create → save picks → load, undo (delete pick), list filtering.

### Acceptance criteria

- `draft_session` and `draft_pick` tables exist in the schema.
- `SqliteDraftSessionRepo` can create a session, save/delete picks, load a full session, and list sessions.
- Loading a session replays picks into a `DraftEngine` that matches the original state.
- A protocol is defined in `protocols.py` for the repo interface.
- All repo methods are covered by tests with an in-memory SQLite database.

## Phase 2: Auto-persist on every mutation

Wire the repo into `DraftSession` so that every pick and undo is immediately written to the database.

### Context

After phase 1, the repo can persist sessions but nothing calls it yet. `DraftSession` currently tracks `_unsaved: bool` and writes JSON on quit. This phase injects the repo and calls it on every state change, making crash recovery automatic. The existing JSON persistence (`save_draft` / `load_draft` / `--resume`) stays for now as a fallback but is no longer the primary path.

### Steps

1. Add a `session_repo` parameter (optional, for backwards compat) to `DraftSession.__init__`. Store `_session_id: int | None`.
2. On session start (in `run()` or a new `_init_session()` helper), call `repo.create_session()` to get a `session_id`. If resuming an existing session, accept the id instead.
3. After each successful `engine.pick()` in `_handle_pick()`, call `repo.save_pick(session_id, pick)`.
4. After each `engine.undo()` in `_handle_undo()`, call `repo.delete_pick(session_id, pick_number)`.
5. On `quit` / `report` (draft complete), call `repo.update_status(session_id, "complete")`.
6. Update the `draft start` CLI command to open a DB connection and inject the repo. Derive the `league` name from the `--league` flag.
7. Update tests: verify repo methods are called on pick/undo/quit using a fake repo injected via constructor.

### Acceptance criteria

- Every `pick` command writes the pick to `draft_pick` immediately (before the next prompt).
- Every `undo` command deletes the undone pick from `draft_pick` immediately.
- If the process is killed after N picks, the database contains exactly those N picks.
- Session status is set to `complete` on normal exit.
- `DraftSession` works without a repo (graceful degradation for tests and mock drafts).
- Existing JSON save/load still works as before.

## Phase 3: Session discovery and resume UX

Make it easy to find and resume interrupted sessions from the CLI.

### Context

With auto-persist in place, the database always has an up-to-date record of every in-progress session. This phase adds CLI affordances so the user doesn't need to remember session IDs or file paths.

### Steps

1. Add a `draft sessions` CLI command that queries `repo.list_sessions()` and displays a table: session ID, league, season, format, pick count (current / total), status, created/updated timestamps.
2. Update `draft start` to check for an existing `in_progress` session for the same `(league, season)`. If found, prompt: "Found in-progress draft (pick N of M). Resume? [Y/n]". On yes, load it from DB; on no, mark the old session `abandoned` and start fresh.
3. Add a `--session-id` option to `draft start` for explicitly resuming a specific session by ID (useful when multiple sessions exist for the same league/season).
4. Add a `draft delete --session-id N` command that deletes a session and its picks (with confirmation).
5. Add tests for the discovery prompt flow and the new CLI commands.

### Acceptance criteria

- `draft sessions` lists all sessions with status, pick progress, and timestamps.
- `draft start` auto-detects and offers to resume an in-progress session for the same league/season.
- `--session-id` allows explicit resume of any session.
- `draft delete` removes a session and its picks with confirmation.
- Old sessions can be marked `abandoned` without data loss (rows stay for analytics).
- Tests cover the prompt flow (resume yes, resume no, no existing session).

## Ordering

Phases are strictly sequential:

1. **Phase 1** (schema + repo) has no dependencies and is pure data-layer work.
2. **Phase 2** (auto-persist) depends on phase 1 — needs the repo to write to.
3. **Phase 3** (discovery UX) depends on phase 2 — needs sessions in the DB to discover.

All phases are independent of other active roadmaps. The `yahoo_draft_pick` table (used by `SqliteYahooDraftRepo` for Yahoo-sourced picks) is a separate concern — it stores raw Yahoo data, while these tables store the user's interactive session state. They can be joined on `player_id` + `season` for cross-analysis.
