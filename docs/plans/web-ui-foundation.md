# Web UI Foundation Roadmap

Stand up a React frontend served by a FastAPI backend that exposes the project's analysis services as a JSON API. This replaces the static HTML export and Flask live-draft server with an interactive single-page application. The foundation provides the shared infrastructure — API layer, React scaffold, build toolchain, navigation shell, and one fully realized view (the draft board with sorting, filtering, and position toggles) — that all future web features (charts, LLM chat, live draft tracker UI) will build on.

The backend is a thin JSON translation layer over `AnalysisContainer`. All business logic stays in the existing service layer; the API serializes domain objects to JSON and the React frontend handles presentation and interactivity. The CLI remains the primary interface for automation and scripting.

## Status

| Phase | Status |
|-------|--------|
| 1 — JSON API | not started |
| 2 — React scaffold and draft board view | not started |
| 3 — API expansion and navigation shell | not started |

## Phase 1: JSON API

Stand up a FastAPI application that serves draft board data as JSON and replaces the Flask live-draft server.

### Context

The existing Flask server (`cli/_live_server.py`) renders HTML server-side and relies on `<meta http-equiv="refresh">` for updates. A JSON API decouples data from presentation, letting the React frontend (phase 2) handle rendering, sorting, and filtering client-side. FastAPI is a better fit than Flask for a JSON API — it provides automatic request/response validation via Pydantic, OpenAPI docs out of the box, and native `async` support for future SSE streaming (LLM chat).

### Steps

1. Add `fastapi` and `uvicorn` as project dependencies.
2. Create a `web/` subpackage under the main package (`src/fantasy_baseball_manager/web/`). Add `app.py` with a FastAPI application factory `create_app(container: AnalysisContainer, league: LeagueSettings) -> FastAPI`.
3. Define Pydantic response models mirroring the domain types: `DraftBoardRowResponse`, `DraftBoardResponse`. These are serialization schemas, not replacements for the domain dataclasses.
4. Implement `GET /api/draft-board` endpoint that accepts query parameters (`season`, `system`, `version`, `player_type`, `position`, `top`) and returns the full draft board as JSON. Reuse `build_draft_board()` internally.
5. Implement `POST /api/draft/{player_id}` and `POST /api/draft/reset` endpoints, migrating the live-draft state management from the Flask server.
6. Implement `GET /api/league` endpoint returning the active league settings (name, categories, roster configuration).
7. Add `fbm web` CLI command that opens the DB, constructs `AnalysisContainer`, and starts the uvicorn server. Accept `--host`, `--port`, `--season`, `--system`, `--version`, `--league` flags.
8. Write tests for all endpoints using FastAPI's `TestClient` with an in-memory SQLite database and seeded data.

### Acceptance criteria

- `GET /api/draft-board` returns a JSON array of player objects with all `DraftBoardRow` fields.
- Query parameters filter the board by player type, position, and top-N.
- `POST /api/draft/{player_id}` removes a player from subsequent `GET /api/draft-board` responses.
- `POST /api/draft/reset` restores the full player pool.
- `GET /api/league` returns league metadata including category names and roster slots.
- All endpoints are covered by tests using `TestClient`.
- FastAPI auto-generated docs are accessible at `/docs`.

## Phase 2: React scaffold and draft board view

Create the React application, dev toolchain, and the interactive draft board table that replaces the static HTML export.

### Context

The draft board is the project's most-used view and the one that prompted this roadmap — it currently lacks client-side sorting, filtering, and combined hitter/pitcher display. Building it first proves the full stack end-to-end (React → FastAPI → AnalysisContainer → SQLite) and establishes the patterns that subsequent views will follow.

### Steps

1. Create a `frontend/` directory at the project root. Initialize a React + TypeScript project using Vite. Add TanStack Table (for sortable/filterable data grids) and a CSS solution (CSS Modules or Tailwind — keep it minimal).
2. Configure Vite's dev server to proxy `/api` requests to the FastAPI backend, so both run during development without CORS issues.
3. Add TypeScript interfaces matching the API's Pydantic response models (`DraftBoardRow`, `DraftBoard`, `LeagueSettings`).
4. Build a `DraftBoardTable` component using TanStack Table:
   - Default view shows all players (hitters and pitchers combined), sorted by rank.
   - All columns are sortable by clicking the header.
   - Position filter control (tabs or dropdown): All, Batters, Pitchers, and each individual position (C, 1B, 2B, SS, 3B, OF, DH, SP, RP).
   - Tier color-coding as row background (matching the existing HTML tier colors).
   - ADP delta highlighting (green for buy, red for avoid).
   - Sticky header row.
5. Add a "Draft" button on each row that POSTs to `/api/draft/{player_id}` and removes the player from the table (for live-draft use).
6. Add a "Reset" button that POSTs to `/api/draft/reset`.
7. Configure FastAPI to serve the Vite production build's static files at `/` (so `fbm web` serves both the API and the frontend from a single process).
8. Update the `fbm web` CLI command to serve the built frontend assets. Document the `npm run build` step (or add a `uv run` script that invokes it).
9. Write component tests for the draft board table (filtering, sorting behavior) using Vitest and Testing Library.

### Acceptance criteria

- `fbm web` starts a single server that serves both the React app at `/` and the API at `/api/*`.
- The draft board table displays all players (hitters and pitchers combined) by default.
- Clicking any column header sorts the table by that column (ascending/descending toggle).
- Position filter controls switch between All, Batters, Pitchers, and individual positions.
- Tier colors and ADP delta highlighting match the existing HTML export's visual design.
- Draft/Reset buttons work and update the table without a full page reload.
- `npm run dev` (frontend) + `fbm web` (backend) work together in development via proxy.
- `npm run build` produces static assets that FastAPI serves in production mode.

## Phase 3: API expansion and navigation shell

Expose additional services as API endpoints and build the app shell (sidebar navigation, multiple views) so the web UI becomes a general-purpose tool beyond just the draft board.

### Context

With the stack proven end-to-end in phase 2, expanding to other views is mostly mechanical: add an API endpoint wrapping an existing service, add a React page consuming it. This phase adds the navigation infrastructure and the next most useful views — projections lookup and valuations — establishing the pattern for all future views. It does not add charts or the LLM chat; those are separate roadmaps building on this foundation.

### Steps

1. Add API endpoints:
   - `GET /api/projections` — player projections (wraps `ProjectionLookupService`). Query params: `season`, `system`, `player_name`.
   - `GET /api/valuations` — player valuations (wraps `ValuationLookupService`). Query params: `season`, `system`, `version`, `player_type`, `position`, `top`.
   - `GET /api/adp/report` — ADP vs value report (wraps `ADPReportService`). Query params: `season`, `system`, `version`, `provider`.
   - `GET /api/players/search` — player search (wraps `PlayerBiographyService.search`). Query params: `name`, `season`.
2. Add Pydantic response models for each new endpoint.
3. Build a navigation shell in React: sidebar or top nav with links to Draft Board, Projections, Valuations, ADP Report, Player Search.
4. Build page components for each new view, each using TanStack Table for tabular data with sorting and filtering.
5. Add a player detail drawer/modal: clicking a player name anywhere in the app shows their bio, projections across systems, and valuations in a side panel. This reuses `GET /api/players/search` and `GET /api/projections`.
6. Write tests for new API endpoints and React components.

### Acceptance criteria

- All new API endpoints return correct JSON and are covered by tests.
- Navigation shell allows switching between views without a full page reload.
- Each view supports sorting and filtering appropriate to its data.
- Clicking a player name opens a detail panel with bio, projections, and valuations.
- The app is usable as a general-purpose analysis tool, not just a draft board.

## Ordering

Phase 1 → 2 → 3, strictly sequential. Phase 1 establishes the API layer that phase 2's React frontend consumes, and phase 3 expands both the API and frontend together. Each phase is independently mergeable — phase 1 is useful on its own (other consumers like the LLM agent could call the JSON API), and phase 2 delivers the original draft board improvements.

This roadmap has no hard dependencies on other roadmaps — it consumes `AnalysisContainer` and the existing services which are already built. It benefits from the LLM Agent roadmap being complete (phase 3 of that roadmap could target the web UI instead of or in addition to the CLI), and it is a prerequisite for any future web-based features (charts, LLM chat panel, live draft tracker UI).
