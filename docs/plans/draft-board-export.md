# Draft Board / Cheat Sheet Export Roadmap

Export a consolidated draft-day artifact that combines valuations, tiers, ADP comparison, and positional eligibility into a single printable or importable document. The goal is a one-stop cheat sheet that a drafter can reference without needing the CLI open during the draft.

This roadmap depends on the valuation system (already built) and optionally integrates data from the tier generator and ADP integration roadmaps when available. The export pipeline is designed to degrade gracefully — tiers and ADP columns are included when the data exists and omitted otherwise.

## Phase 1: Draft board service

Build a stateless draft board service that combines valuations + league settings (and optionally tiers/ADP) into a ranked board of players. This is the core data structure that all draft-day consumers (CLI, HTML, TUI, agents, live draft tracker) will build on. No CLI, no file I/O — just the service layer and tests.

- [Phase 1 plan](draft-board-export/phase-1.md)

### Acceptance criteria

- `DraftBoard` / `DraftBoardRow` frozen dataclasses expose all necessary fields.
- `build_draft_board()` is a pure function accepting the player pool, league settings, and optional enrichment data.
- Category z-scores are filtered to the league's configured categories per player type.
- Tier and ADP enrichment are joined when provided, None when absent.
- Two-way player ADP resolution matches pitcher/batter context.
- Board is re-callable with a shrinking pool (simulating live draft picks).

## Phase 2: CLI display & CSV export

Add terminal display (`fbm draft board`) and CSV/TSV export (`fbm draft export`) commands.

### Context

Many drafters use Google Sheets, Excel, or platform-specific import tools (ESPN, Yahoo) that accept CSV. A CLI display for quick checks and a CSV export for spreadsheet import covers the majority of use cases.

### Steps

1. Implement `export_csv(board, path)` that writes the board to a CSV file. Include a header row. Omit columns that are entirely None (e.g., tier and ADP columns when those systems haven't been run).
2. Add `fbm draft board` CLI command for terminal display (rich table).
3. Add `fbm draft export --season <year> --system <system> --format csv --output <path>` CLI command.
4. Write tests for CSV output and CLI integration.

### Acceptance criteria

- CSV export includes all valuation data with correct column headers.
- Tier and ADP columns appear when data exists, are omitted when not.
- File is importable into Google Sheets / Excel without formatting issues.
- Player positions use standard abbreviations (C, 1B, 2B, SS, 3B, OF, SP, RP).

## Phase 3: HTML rendering + Flask live server

Generate a formatted, printable HTML cheat sheet with color-coded tiers and visual highlights for value picks. Optionally serve via Flask for live draft updates.

### Context

For drafters who want a printed reference or a browser tab open during the draft, a styled HTML page is more scannable than a spreadsheet. Color-coding tiers and highlighting value-over-ADP outliers makes the key information pop.

### Steps

1. Add `export_html(board, league_settings, path)` that renders the draft board as a styled HTML table using string templates.
2. Color-code tier bands (alternating background colors per tier).
3. Highlight players where `adp_delta > threshold` in green (buy) and red (avoid).
4. Group by player type (batters first, then pitchers), with position subgroups within each.
5. Include a print-friendly CSS stylesheet (no web fonts, compact margins, page-break rules).
6. Add `--format html` option to the `fbm draft export` command.
7. Add optional Flask live server that re-renders on pool changes.
8. Write tests verifying HTML structure contains expected elements.

### Acceptance criteria

- HTML file renders correctly in a browser.
- Tiers are visually distinguishable via color coding.
- Value outliers are highlighted.
- Prints cleanly on letter/A4 paper (landscape orientation).

## Status

| Phase | Status |
|-------|--------|
| 1 — Draft board service | done (2026-02-17) |
| 2 — CLI display & CSV export | done (2026-02-18) |
| 3 — HTML rendering + Flask live server | done (2026-02-20) |

## Ordering

Phase 1 -> 2 -> 3, sequential. Phase 1 is the core data layer and must land first. Phase 2 adds user-facing output. Phase 3 adds polish and live capabilities. This roadmap can begin immediately (phase 1 only needs valuations) but benefits from the tier generator and ADP integration being completed first.
