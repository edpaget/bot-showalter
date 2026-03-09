# Web UI Foundation Roadmap

Build a GraphQL API (Strawberry + FastAPI) and React frontend for the fantasy baseball manager, with live draft as the primary use case. The API exposes the project's analysis services as a typed GraphQL schema; the React frontend provides an interactive draft-day dashboard that replaces both the CLI REPL and the auto-refreshing HTML board.

The backend is a thin GraphQL translation layer over `AnalysisContainer` and `DraftEngine`. All business logic stays in the existing service layer; Strawberry resolvers map domain dataclasses to GraphQL types and delegate to services. The CLI remains the primary interface for automation and scripting.

The UI design specification is at [`docs/draft-ui-design.md`](../draft-ui-design.md).

## Status

| Phase | Status |
|-------|--------|
| 1 — GraphQL schema and read-only queries | done (2026-03-07) |
| 2 — Session mutations and queries | done (2026-03-07) |
| 3 — Subscriptions and Yahoo polling | done (2026-03-07) |
| 4 — React scaffold and draft board view | in progress |
| 5 — Live draft dashboard | not started |
| 6 — Analysis views and navigation | not started |

## Phase 1: GraphQL schema and read-only queries

Stand up a Strawberry GraphQL API on FastAPI that serves draft board data, tiers, scarcity, and league settings as read-only queries.

### Context

The existing Flask server (`cli/_live_server.py`) renders HTML server-side with `<meta http-equiv="refresh">` for updates. A GraphQL API decouples data from presentation and gives the React frontend (phase 4) precise control over what it fetches. Strawberry's dataclass-style type definitions map naturally onto the existing domain objects (`DraftBoardRow`, `DraftBoard`, `LeagueSettings`, etc.), minimizing boilerplate. FastAPI provides the HTTP layer, automatic OpenAPI docs, and Strawberry's `GraphQLRouter` integration.

### Steps

1. Add `strawberry-graphql[fastapi]`, `fastapi`, and `uvicorn` as project dependencies.
2. Create `src/fantasy_baseball_manager/web/` subpackage. Add `app.py` with a FastAPI application factory `create_app(container: AnalysisContainer, league: LeagueSettings) -> FastAPI` that mounts Strawberry's `GraphQLRouter` at `/graphql`.
3. Define Strawberry types mirroring the domain dataclasses: `DraftBoardRowType`, `DraftBoardType`, `LeagueSettingsType`, `TierAssignmentType`, `ScarcityReportType`. Use Strawberry's `@strawberry.type` decorator — these are serialization types, not replacements for domain dataclasses. `DraftBoardRowType` must include breakout/bust rank fields (integer rank within player type, e.g., 3 = third-highest breakout candidate among batters).
4. Implement a `Query` root type with resolvers:
   - `board(season, system, version, player_type, position, top) -> DraftBoardType` — wraps `build_draft_board()`, enriched with breakout/bust ranks from the classifier model.
   - `tiers(season, system, version, player_type, method) -> list[TierAssignmentType]` — wraps tier generation.
   - `scarcity(season, system, version) -> list[ScarcityReportType]` — wraps scarcity analysis.
   - `league() -> LeagueSettingsType` — returns the active league configuration.
5. Add `fbm web` CLI command that opens the DB, constructs `AnalysisContainer`, and starts the uvicorn server. Accept `--host`, `--port`, `--season`, `--system`, `--version`, `--league` flags.
6. Write tests for all resolvers using Strawberry's test client with an in-memory SQLite database and seeded data.

### Acceptance criteria

- `fbm web` starts a server with the GraphQL playground accessible at `/graphql`.
- Board query returns all `DraftBoardRow` fields with correct filtering by player type, position, and top-N.
- Tiers query returns tier assignments grouped by position.
- Scarcity query returns per-position scarcity metrics.
- League query returns category names, roster slots, and league format.
- All resolvers are covered by tests.

## Phase 2: Session mutations and queries

Add stateful draft session management — start a session, record picks, undo, and query recommendations, roster, needs, and category balance through GraphQL. Session state is persisted to SQLite on every mutation via `SqliteDraftSessionRepo` (from the [draft-session-persistence](draft-session-persistence.md) roadmap), making sessions crash-safe and resumable.

### Context

The live draft tracker's core value is the `DraftEngine` (state management) and `recommend()` (pick suggestions). These are currently only accessible via the CLI REPL in `DraftSession`. Exposing them as GraphQL mutations and queries lets the React frontend drive a draft session with the same capabilities. Mutations naturally return the updated state, so the frontend can refresh all panels (recommendations, roster, needs, balance) in a single round trip after each pick.

Session persistence uses `SqliteDraftSessionRepo` rather than JSON files. Every pick and undo is written to the database immediately, so if the browser closes or the server crashes, the session can be resumed from exactly where it left off. This also enables a `sessions` query for listing and resuming past drafts.

### Steps

1. Define Strawberry types for session state: `DraftPickType`, `DraftStateType`, `DraftSessionSummaryType`, `RecommendationType`, `RosterSlotType`, `CategoryBalanceType`.
2. Create a session manager class that maintains a cache of hydrated `DraftEngine` instances keyed by session ID (`dict[int, DraftEngine]`). When a resolver requests a session ID that isn't in the cache, the manager hydrates it from SQLite by loading the config and replaying picks. Inject `SqliteDraftSessionRepo` for persistence. Store the manager in Strawberry's context so resolvers can access it. Initially only one session will be active at a time, but the `sessionId`-keyed design keeps the door open for concurrent sessions later.
3. Implement mutations (all scoped by `sessionId`):
   - `startSession(league, season, system, teams, slot, format) -> DraftStateType` — creates a new session via `repo.create_session()`, initializes `DraftEngine.start()`, caches the engine, and returns the state including the new session ID.
   - `pick(sessionId, playerId, position, price) -> PickResultType` — records a pick via `DraftEngine.pick()`, persists it via `repo.save_pick()`, returns the pick plus updated recommendations, roster, and needs.
   - `undo(sessionId) -> PickResultType` — reverses the last pick via `DraftEngine.undo()`, deletes it via `repo.delete_pick()`, returns updated state.
   - `endSession(sessionId) -> Boolean` — marks the session `complete` via `repo.update_status()` and evicts it from the cache.
4. Implement session queries (all scoped by `sessionId` where applicable):
   - `session(sessionId) -> DraftStateType` — session state (picks, current pick, team on clock). Hydrates the engine from SQLite on first access if not already cached.
   - `sessions(league, season, status) -> list[DraftSessionSummaryType]` — lists past and in-progress sessions with metadata (ID, league, season, format, pick count, status, timestamps). Wraps `repo.list_sessions()`.
   - `recommendations(sessionId, position, limit) -> list[RecommendationType]` — wraps `recommend()` with category balance integration.
   - `roster(sessionId, team) -> list[DraftPickType]` — team roster (defaults to user's team).
   - `needs(sessionId) -> list[RosterSlotType]` — unfilled positions with slot counts.
   - `balance(sessionId) -> list[CategoryBalanceType]` — category projections and strength rankings.
   - `available(sessionId, position, limit) -> list[DraftBoardRowType]` — remaining player pool.
5. Define a `PickResultType` that bundles the pick confirmation with recommendations, roster, needs, and balance — enabling the frontend to update all panels from one mutation response.
6. Write tests: start session, query by session ID (cache miss triggers hydration from DB), pick/undo sequences with repo verification, recommendation changes as roster fills, session listing, crash recovery (kill server, restart, query same session ID and verify it hydrates correctly).

### Acceptance criteria

- All mutations and session-specific queries accept a `sessionId` parameter.
- Querying a session ID that exists in SQLite but not in the in-memory cache hydrates it automatically (replays picks into a `DraftEngine`).
- `startSession` mutation creates a new session and returns its ID.
- `pick` mutation validates roster constraints and budget limits, persists the pick to SQLite immediately, and returns updated recommendations and needs.
- `undo` mutation fully reverses the last pick (player returns to pool, budget restored) and deletes the pick from SQLite.
- `sessions` query returns past and in-progress sessions with pick counts and timestamps.
- `endSession` marks the session complete in the database.
- If the server crashes after N picks, querying the same session ID on restart hydrates from SQLite at pick N.
- `recommendations` query reflects current roster state — unfilled positions are boosted, scarcity is factored in.
- `PickResultType` bundles all panel data so the frontend can update in one round trip.

## Phase 3: Subscriptions and Yahoo polling

Add GraphQL subscriptions for real-time draft events and integrate the Yahoo draft poller so opponent picks appear automatically.

### Context

During a live Yahoo draft, opponents' picks arrive via the Yahoo API poller (`YahooDraftPoller`). Currently the CLI REPL drains a queue and prints updates. With GraphQL subscriptions over WebSocket, the React frontend receives pick events in real time without polling. This is the key upgrade over the auto-refreshing HTML board — the UI updates instantly when any pick happens.

### Steps

1. Define a `DraftEventType` union: `PickEvent` (a pick was made), `UndoEvent` (a pick was reversed), `SessionEvent` (session started/ended).
2. Implement a Strawberry subscription `draftEvents(sessionId) -> AsyncGenerator[DraftEventType]` that yields events for a specific session. Use a per-session `asyncio.Queue` as the event bus between mutations/poller and the subscription resolver.
3. Wire mutations (`pick`, `undo`, `startSession`, `endSession`) to publish events to the session's subscription queue.
4. Add mutations for Yahoo polling control:
   - `startYahooPoll(sessionId, leagueKey) -> Boolean` — starts `YahooDraftPoller` in a background thread, translates incoming picks via `ingest_yahoo_pick()`, and publishes them as `PickEvent`s on the session's queue.
   - `stopYahooPoll(sessionId) -> Boolean` — stops the poller.
5. Add a `yahooPollStatus` query returning whether polling is active and the last poll timestamp.
6. Configure Strawberry's WebSocket integration with FastAPI for subscription transport.
7. Write tests: subscription receives events from mutations, Yahoo poller events propagate to subscription, poller start/stop lifecycle.

### Acceptance criteria

- GraphQL subscription delivers `PickEvent` in real time when a pick mutation is called.
- Yahoo poller picks appear as subscription events without manual intervention.
- `startYahooPoll` / `stopYahooPoll` control the poller lifecycle.
- Subscription reconnects cleanly if the WebSocket drops.
- Poller thread shuts down when the session ends or the server stops.

## Phase 4: React scaffold and draft board view

Create the React application with Apollo Client and build the interactive draft board table that replaces the static HTML export.

### Context

The draft board is the project's most-used view. It currently lacks client-side sorting, filtering, and combined hitter/pitcher display. This phase proves the full stack end-to-end (React + Apollo Client -> Strawberry -> AnalysisContainer -> SQLite) and establishes the component patterns that the live draft dashboard (phase 5) builds on.

### Steps

1. Create a `frontend/` directory at the project root. Initialize a React + TypeScript project using Vite with bun as the package manager. Add Apollo Client for GraphQL and Tailwind CSS.
2. Configure Vite's dev server to proxy `/graphql` requests to the FastAPI backend.
3. Define GraphQL queries in `.graphql` files or `gql` tagged templates for the board, tiers, and league.
4. Build a `DraftBoardTable` component (see [`docs/draft-ui-design.md`](../draft-ui-design.md) for column spec and styling):
   - Columns: Rank, Player, Pos, Tier, Value, ADP, ADP Delta, Breakout, Bust, Status, Action.
   - Default view shows all players (hitters and pitchers combined), sorted by rank.
   - All columns are sortable by clicking the header.
   - Position filter controls: All, Batters, Pitchers, and each individual position.
   - Player type tabs: All / Batters / Pitchers.
   - Status filter: Available / All / Drafted.
   - Tier color-coding as row background (8-color rotation).
   - Breakout/bust display: rank-based labels (e.g., "B3"), top-20 green/red row tinting, tooltip with model description. No raw probabilities.
   - ADP delta highlighting (green for value, red for reach).
   - Drafted players grayed out but visible (filterable).
   - Sticky header row.
5. Add a search/filter input for player name filtering.
6. Configure FastAPI to serve the Vite production build's static files at `/` so `fbm web` serves both the API and frontend from a single process.
7. Update `fbm web` to serve the built frontend assets. Document the build step.
8. Write component tests using Vitest and Testing Library with mocked Apollo responses.

### Acceptance criteria

- `fbm web` starts a single server that serves the React app at `/` and the GraphQL API at `/graphql`.
- The draft board table displays all players by default with sorting on every column.
- Position filter controls switch between All, Batters, Pitchers, and individual positions.
- Tier colors and ADP delta highlighting match the existing HTML export.
- Breakout/bust ranks display as labels (e.g., "B3") with top-20 row tinting and tooltips.
- Player name search filters the table client-side.
- `bun dev` (frontend) + `fbm web` (backend) work together in development via proxy.
- `bun run build` produces static assets that FastAPI serves in production mode.

## Phase 5: Live draft dashboard

Build the full draft-day interface: recommendation panel, roster tracker, needs display, category balance visualization, and pick log — all wired to the GraphQL session mutations and subscriptions.

### Context

This is the capstone phase — the reason the roadmap exists. It replaces both the CLI REPL and the auto-refreshing HTML board with a single interactive dashboard. The frontend consumes the session mutations (phase 2) and subscriptions (phase 3) to provide real-time feedback after every pick. The draft board from phase 4 becomes one panel in a larger layout.

### Steps

1. Build the dashboard layout: draft board (left/main), sidebar with recommendations + roster + needs + balance panels. Pick log at the bottom or in a collapsible panel.
2. Build a `RecommendationPanel` component showing top 5-10 recommendations with scores, positions, values, and reasons. Supports position filtering. Refreshes from `pick` mutation response.
3. Build a `RosterPanel` component showing the user's roster organized by position slot, with filled/empty indicators and total value.
4. Build a `NeedsPanel` component showing unfilled positions with remaining slot counts.
5. Build a `CategoryBalancePanel` component visualizing category strengths — bar chart or radar chart showing z-score totals per category with league rank estimates.
6. Build a `PickLogPanel` component showing the draft history (all teams' picks in order) with the most recent pick highlighted.
7. Add pick interaction: "Draft" button on board rows and recommendation rows triggers the `pick` mutation. Show a confirmation if the pick is unusual (e.g., drafting a position that's already filled).
8. Add undo button that triggers the `undo` mutation.
9. Wire Apollo Client subscriptions so Yahoo-polled picks update all panels automatically.
10. Add session controls: start session (with league/format/teams/slot config), resume session (from session list), and end session. No explicit save button — persistence is automatic via the repo on every pick.
11. Write integration tests: start session, pick sequence, verify panels update, subscription events propagate.

### Acceptance criteria

- Dashboard displays board, recommendations, roster, needs, and balance in a single view.
- Picking a player (via button or recommendation) triggers the `pick` mutation and updates all panels from the response without separate fetches.
- Undo reverses the last pick and updates all panels.
- Yahoo subscription events update the board and pick log in real time.
- Category balance visualization shows relative strengths across all categories.
- Session can be started, resumed from the session list, and ended. Persistence is automatic — no manual save step.
- Dashboard is usable under draft-day time pressure — interactions are fast, layout is clear, no unnecessary clicks.

## Phase 6: Analysis views and navigation

Expand the GraphQL schema with projections, valuations, and player search, and add a navigation shell so the web UI is useful beyond draft day.

### Context

With the draft dashboard complete, expanding to other views is mostly mechanical: add a resolver wrapping an existing service, add a React page consuming it. This phase adds navigation infrastructure and the next most useful views — player lookup, projections comparison, and valuations — establishing the pattern for future views.

### Steps

1. Add GraphQL queries:
   - `projections(season, system, playerName) -> list[ProjectionType]` — wraps `ProjectionLookupService`.
   - `valuations(season, system, version, playerType, position, top) -> list[ValuationType]` — wraps `ValuationLookupService`.
   - `adpReport(season, system, version, provider) -> list[ADPReportRowType]` — wraps ADP report service.
   - `playerSearch(name, season) -> list[PlayerType]` — wraps player biography search.
2. Define Strawberry types for each new query's response.
3. Build a navigation shell: sidebar or top nav with links to Draft Dashboard, Draft Board, Projections, Valuations, ADP Report, Player Search.
4. Build page components for each new view using a sortable/filterable table pattern.
5. Add a player detail drawer: clicking a player name anywhere opens a side panel with bio, projections across systems, and valuations.
6. Write tests for new resolvers and React components.

### Acceptance criteria

- All new queries return correct data and are covered by tests.
- Navigation shell allows switching between views without a full page reload.
- Each view supports sorting and filtering.
- Clicking a player name opens a detail panel with bio, projections, and valuations.
- The app is usable as a general-purpose analysis tool beyond draft day.

## Ordering

Phases 1 through 3 are the backend, phases 4 through 6 are the frontend. Within each tier, phases are sequential:

- **Phase 1 -> 2 -> 3** (backend): Phase 1 establishes the schema and read-only queries. Phase 2 adds session state. Phase 3 adds real-time subscriptions.
- **Phase 4 -> 5 -> 6** (frontend): Phase 4 builds the React scaffold and board view. Phase 5 adds the live draft panels. Phase 6 expands to analysis views.

Phases 4 can start as soon as phase 1 is complete (it only needs read-only queries). Phase 5 requires phases 2 and 3 (it needs mutations and subscriptions). Phase 6 only depends on phase 4 (navigation shell).

The critical path for draft day is: **1 -> 2 -> 3 -> 4 -> 5**. Phase 6 is a post-draft-season enhancement.

Phase 2 depends on the [draft-session-persistence](draft-session-persistence.md) roadmap (at least phases 1-2) for `SqliteDraftSessionRepo` and the `draft_session` / `draft_pick` tables. If draft-session-persistence is not yet complete when phase 2 begins, its first two phases should be implemented first.

This roadmap replaces the Flask live-draft server (`cli/_live_server.py`) and the static HTML export with a richer, interactive alternative. The CLI remains fully functional.
