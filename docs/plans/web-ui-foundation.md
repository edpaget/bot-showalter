# Web UI Foundation Roadmap

Build a GraphQL API (Strawberry + FastAPI) and React frontend for the fantasy baseball manager, with live draft as the primary use case. The API exposes the project's analysis services as a typed GraphQL schema; the React frontend provides an interactive draft-day dashboard that replaces both the CLI REPL and the auto-refreshing HTML board.

The backend is a thin GraphQL translation layer over `AnalysisContainer` and `DraftEngine`. All business logic stays in the existing service layer; Strawberry resolvers map domain dataclasses to GraphQL types and delegate to services. The CLI remains the primary interface for automation and scripting.

The UI design specification is at [`docs/draft-ui-design.md`](../draft-ui-design.md).

## Status

| Phase | Status |
|-------|--------|
| 1 — GraphQL schema and read-only queries | not started |
| 2 — Session mutations and queries | not started |
| 3 — Subscriptions and Yahoo polling | not started |
| 4 — React scaffold and draft board view | not started |
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

Add stateful draft session management — start a session, record picks, undo, and query recommendations, roster, needs, and category balance through GraphQL.

### Context

The live draft tracker's core value is the `DraftEngine` (state management) and `recommend()` (pick suggestions). These are currently only accessible via the CLI REPL in `DraftSession`. Exposing them as GraphQL mutations and queries lets the React frontend drive a draft session with the same capabilities. Mutations naturally return the updated state, so the frontend can refresh all panels (recommendations, roster, needs, balance) in a single round trip after each pick.

### Steps

1. Define Strawberry types for session state: `DraftPickType`, `DraftStateType`, `RecommendationType`, `RosterSlotType`, `CategoryBalanceType`.
2. Create a session manager class that holds the active `DraftEngine` and `DraftState` in memory (one session at a time — this is a single-user tool). Store it in Strawberry's context so resolvers can access it.
3. Implement mutations:
   - `startSession(season, system, teams, slot, format) -> DraftStateType` — initializes `DraftEngine.start()` with the full player pool.
   - `pick(playerId, position, price) -> PickResultType` — records a pick via `DraftEngine.pick()`, returns the pick plus updated recommendations, roster, and needs.
   - `undo() -> PickResultType` — reverses the last pick, returns updated state.
   - `saveSession(name) -> Boolean` — persists draft state to JSON.
   - `loadSession(name) -> DraftStateType` — restores a saved session.
4. Implement session queries:
   - `session -> DraftStateType` — current session state (picks, current pick, team on clock).
   - `recommendations(position, limit) -> list[RecommendationType]` — wraps `recommend()` with category balance integration.
   - `roster(team) -> list[DraftPickType]` — team roster (defaults to user's team).
   - `needs() -> list[RosterSlotType]` — unfilled positions with slot counts.
   - `balance() -> list[CategoryBalanceType]` — category projections and strength rankings.
   - `available(position, limit) -> list[DraftBoardRowType]` — remaining player pool.
5. Define a `PickResultType` that bundles the pick confirmation with recommendations, roster, needs, and balance — enabling the frontend to update all panels from one mutation response.
6. Write tests: start session, pick/undo sequences, recommendation changes as roster fills, save/load round-trip.

### Acceptance criteria

- `startSession` mutation initializes a draft with the correct player pool and snake/auction configuration.
- `pick` mutation validates roster constraints and budget limits, returns the pick plus updated recommendations and needs.
- `undo` mutation fully reverses the last pick (player returns to pool, budget restored).
- `recommendations` query reflects current roster state — unfilled positions are boosted, scarcity is factored in.
- `saveSession` and `loadSession` round-trip correctly.
- `PickResultType` bundles all panel data so the frontend can update in one round trip.

## Phase 3: Subscriptions and Yahoo polling

Add GraphQL subscriptions for real-time draft events and integrate the Yahoo draft poller so opponent picks appear automatically.

### Context

During a live Yahoo draft, opponents' picks arrive via the Yahoo API poller (`YahooDraftPoller`). Currently the CLI REPL drains a queue and prints updates. With GraphQL subscriptions over WebSocket, the React frontend receives pick events in real time without polling. This is the key upgrade over the auto-refreshing HTML board — the UI updates instantly when any pick happens.

### Steps

1. Define a `DraftEventType` union: `PickEvent` (a pick was made), `UndoEvent` (a pick was reversed), `SessionEvent` (session started/loaded/saved).
2. Implement a Strawberry subscription `draftEvents() -> AsyncGenerator[DraftEventType]` that yields events as they occur. Use an `asyncio.Queue` as the event bus between mutations/poller and the subscription resolver.
3. Wire mutations (`pick`, `undo`, `startSession`) to publish events to the subscription queue.
4. Add mutations for Yahoo polling control:
   - `startYahooPoll(leagueKey) -> Boolean` — starts `YahooDraftPoller` in a background thread, translates incoming picks via `ingest_yahoo_pick()`, and publishes them as `PickEvent`s.
   - `stopYahooPoll() -> Boolean` — stops the poller.
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

1. Create a `frontend/` directory at the project root. Initialize a React + TypeScript project using Vite. Add Apollo Client for GraphQL and a CSS solution (Tailwind or CSS Modules).
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
- `npm run dev` (frontend) + `fbm web` (backend) work together in development via proxy.
- `npm run build` produces static assets that FastAPI serves in production mode.

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
10. Add session controls: start session (with format/teams/slot config), save, load, and end session.
11. Write integration tests: start session, pick sequence, verify panels update, subscription events propagate.

### Acceptance criteria

- Dashboard displays board, recommendations, roster, needs, and balance in a single view.
- Picking a player (via button or recommendation) triggers the `pick` mutation and updates all panels from the response without separate fetches.
- Undo reverses the last pick and updates all panels.
- Yahoo subscription events update the board and pick log in real time.
- Category balance visualization shows relative strengths across all categories.
- Session can be started, saved, loaded, and resumed.
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

This roadmap has no hard dependencies on other roadmaps — it consumes `AnalysisContainer` and the existing services which are already built. It replaces the Flask live-draft server (`cli/_live_server.py`) and the static HTML export with a richer, interactive alternative. The CLI remains fully functional.
