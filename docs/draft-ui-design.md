# Draft UI Design

Design specification for the web-based draft dashboard. The implementation roadmap is at [`docs/plans/web-ui-foundation.md`](plans/web-ui-foundation.md).

## Architecture

React + TypeScript frontend communicating with a Strawberry GraphQL API over FastAPI. Apollo Client handles queries, mutations, and WebSocket subscriptions. The backend is a thin resolver layer over existing services (`AnalysisContainer`, `DraftEngine`, `recommend()`). A single `fbm web` command serves both the API and frontend.

## Dashboard Layout

The draft dashboard is a single-page view with three zones:

```
+--------------------------------------------------+
|  Controls: session config, save/load, Yahoo poll  |
+---------------------------+----------------------+
|                           |  Recommendations     |
|   Draft Board Table       |  (top 5-10, scores,  |
|   (main, scrollable)      |   reasons)           |
|                           +----------------------+
|                           |  My Roster           |
|                           |  (by position slot,  |
|                           |   budget remaining)  |
|                           +----------------------+
|                           |  Category Balance    |
|                           |  (bar/radar chart)   |
+---------------------------+----------------------+
|  Pick Log (collapsible, most recent highlighted)  |
+--------------------------------------------------+
```

## Draft Board Table

The central element is a sortable, filterable table of all draftable players.

### Columns

| Column | Description |
|--------|-------------|
| Rank | Overall rank by auction value (ZAR) |
| Player | Name, team |
| Pos | Eligible positions |
| Tier | Tier assignment (color-coded row background) |
| Value | Projected auction value ($) |
| ADP | Average draft position (overall pick) |
| ADP Delta | Value rank minus ADP rank — positive means undervalued by ADP |
| Breakout | Breakout rank among player type (e.g., "B3" = 3rd-highest breakout candidate among batters). Top-20 highlighted green. |
| Bust | Bust rank among player type (e.g., "X5"). Top-20 highlighted red. |
| Status | Available / Drafted (by whom, for how much in auction) |
| Action | "Draft" button (triggers `pick` mutation) |

### Row Styling

- **Tier color-coding** — row background color by tier assignment (8-color rotation matching existing HTML export)
- **Breakout highlight** — green tint on rows for top-20 breakout candidates (overrides tier color)
- **Bust highlight** — red tint on rows for top-20 bust risks (overrides tier color)
- **Dual signal** — players flagged as both breakout and bust get a split indicator (green/red left-right or diagonal)
- **Drafted players** — grayed out but remain visible; filterable via status control

### Filters and Controls

- **Position filter** — All, C, 1B, 2B, SS, 3B, OF, SP, RP, UTIL
- **Player type tabs** — All / Batters / Pitchers
- **Status filter** — Available only / All / Drafted only
- **Search** — filter by player name (client-side)
- **Sort** — click any column header; default sort by Value descending
- **Sticky header row** — column headers remain visible while scrolling

## Sidebar Panels

### Recommendation Panel

Top 5-10 recommendations from `recommend()`, updated after every pick. Each entry shows:
- Player name, position, value
- Composite score (0-10)
- Reason text (e.g., "fills need at SS + positional scarcity")
- "Draft" button for quick pick

Supports position filtering (e.g., show only SS recommendations).

### Roster Panel

User's roster organized by position slot:
- Filled slots show player name, position, and price paid
- Empty slots highlighted as needs
- Remaining budget (auction) or remaining picks (snake)
- Total roster value

### Category Balance Panel

Visualization of category strengths from projected stats of drafted players:
- Bar chart or radar chart showing z-score totals per category
- League rank estimates per category
- Updates after each pick via mutation response

## Breakout/Bust Signal Display

The breakout/bust model provides **ranking signal only** — the probability values are not meaningful in absolute terms. Display rules:

- **Rank-based labels** rather than probabilities (e.g., "B5" not "42%")
- **Top-20 threshold** for visual highlighting — this is where the model has demonstrated lift
- **Tooltip on hover** showing: "Ranked #5 breakout candidate among batters. Model identifies players likely to outperform ADP but does not predict by how much."
- **No dollar adjustment** — the model doesn't have reliable dollar-value signal

## Pick Log

Collapsible panel at the bottom showing all picks in draft order:
- Pick number, team, player, position, price (auction)
- Most recent pick highlighted
- Updates in real time via GraphQL subscription (including Yahoo-polled opponent picks)

## Session Controls

Top bar with:
- **Start session** — configure league, format (snake/auction), teams, user slot, season, system. Returns a session ID that the frontend stores and passes on all subsequent queries and mutations.
- **Session picker** — dropdown or list of past/in-progress sessions (shows league, season, pick count, timestamps). Selecting a session ID loads it — the backend hydrates the `DraftEngine` from SQLite on first access. Uses `SqliteDraftSessionRepo` for crash-safe persistence — every pick is written to SQLite immediately, so there is no explicit "save" action.
- **Yahoo poll** — start/stop Yahoo draft polling with connection status indicator
- **Undo** — reverse last pick
- **End session** — mark session complete and show post-draft report

## Future Considerations

- Projected standings integration (how does adding this player change my category ranks?)
- Keeper cost overlay for keeper/dynasty leagues
- Mock draft mode for pre-draft practice
- Player detail drawer (click name to see bio, projections across systems, valuations)
