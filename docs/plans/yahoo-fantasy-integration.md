# Yahoo Fantasy API Integration Roadmap

Integrate Yahoo Fantasy Sports API to make the system league-aware — knowing your rosters, opponents, draft picks, and league history across two Yahoo leagues (one keeper, one redraft). This is a foundational layer that replaces manual configuration with live league data and unlocks downstream features like sit/start, trade finders, and waiver wire rankings.

Yahoo's Fantasy Sports REST API (OAuth2, XML/JSON) exposes league metadata, rosters, draft results, transactions, and standings via a resource hierarchy: game → league → team → roster/players. Player identity uses Yahoo-specific IDs that must be crosswalked to our MLBAM-based system. Two-way players (e.g., Ohtani) are represented as separate Yahoo player IDs for batting and pitching, requiring a many-to-one mapping.

## Status

| Phase | Status |
|-------|--------|
| 1 — OAuth client and league metadata sync | not started |
| 2 — Player ID crosswalk and roster ingest | not started |
| 3 — Draft results and live draft tracking | not started |
| 4 — League history and keeper evaluation | not started |
| 5 — Transaction log and league activity | not started |

## Phase 1: OAuth client and league metadata sync

Stand up the Yahoo API client with persistent OAuth2 tokens, fetch league metadata, and auto-populate league settings.

### Context

The system currently requires manual league configuration in `fbm.toml`. Yahoo's API can provide this automatically — scoring categories, roster slots, team count, draft type — and keep it in sync. This phase establishes the HTTP client and auth flow that all subsequent phases depend on.

Yahoo uses OAuth2 with a 3-legged flow: browser redirect → user authorization → access token + refresh token. The user already has Yahoo OAuth app credentials (client_id and client_secret).

### Steps

1. Add `yahoo` config section to `fbm.toml` with `client_id`, `client_secret`, and per-league mappings (league name → Yahoo game key + league ID). Support `FBM_YAHOO_LEAGUE` env var and a `default_league` config key for selecting the active league without `--league` on every command.
2. Create `src/fantasy_baseball_manager/yahoo/auth.py`:
   - Browser-based OAuth2 flow using `httpx` and Python's `webbrowser` module. Open browser to Yahoo's authorization URL, run a tiny local HTTP server to capture the redirect callback.
   - Persist tokens to `~/.config/fbm/yahoo_tokens.json` (per-app, not per-league — one Yahoo user auth covers all leagues).
   - Auto-refresh expired access tokens using the stored refresh token. Token refresh is transparent to callers.
   - `YahooAuth` class with `get_access_token() -> str` that handles the full lifecycle (load from disk → refresh if expired → browser flow if no token).
3. Create `src/fantasy_baseball_manager/yahoo/client.py`:
   - `YahooFantasyClient` class wrapping `httpx.Client` with base URL `https://fantasysports.yahooapis.com/fantasy/v2/`, auth header injection, JSON format parameter, and retry via `tenacity` (matching existing ingest patterns).
   - Methods for the core resources: `get_league(league_key)`, `get_league_settings(league_key)`, `get_teams(league_key)`, `get_roster(team_key)`, `get_draft_results(league_key)`, `get_transactions(league_key)`, `get_players(league_key, player_keys)`.
   - League key format: `{game_key}.l.{league_id}` (e.g., `mlb.l.12345` or `449.l.12345` where 449 is the MLB game ID for a given season).
4. Create `src/fantasy_baseball_manager/yahoo/league_source.py` implementing the `DataSource` protocol:
   - `source_type = "yahoo_league"`, `source_detail` = league key.
   - `fetch()` returns league settings (categories, roster positions, team count, format, draft type, season) as a list of dicts.
5. Create domain types for Yahoo league context in `src/fantasy_baseball_manager/domain/yahoo_league.py`:
   - `YahooLeague` frozen dataclass: `league_key`, `name`, `season`, `num_teams`, `draft_type` ("live_auction" | "live_snake" | "autopick"), `is_keeper`, `game_key`.
   - `YahooTeam` frozen dataclass: `team_key`, `league_key`, `team_id`, `name`, `manager_name`, `is_owned_by_user`.
6. Add migration `015_yahoo_league.sql` for `yahoo_league` and `yahoo_team` tables.
7. Add `fbm yahoo auth` CLI command to trigger the browser-based OAuth flow and confirm token storage.
8. Add `fbm yahoo sync --league <name>` CLI command that fetches league metadata from Yahoo and either validates it matches the `fbm.toml` league config or reports diffs. Print a summary: team count, categories, roster slots, draft type.
9. Write tests: auth token lifecycle (mock HTTP), client request construction, league source fetch/parse, config loading with yahoo section.

### Acceptance criteria

- `fbm yahoo auth` opens a browser, completes OAuth flow, and persists tokens.
- Subsequent API calls use the stored token without re-authenticating.
- Expired tokens are silently refreshed.
- `fbm yahoo sync` fetches and displays league metadata from Yahoo.
- League key is correctly constructed from config (game key + league ID).
- Default league is resolved from env var → toml config → error if neither set.

## Phase 2: Player ID crosswalk and roster ingest

Map Yahoo player IDs to our internal MLBAM-based player IDs and ingest current rosters for all teams.

### Context

Yahoo uses its own player ID system. Every downstream feature (draft tracking, keeper eval, sit/start) requires mapping Yahoo player IDs to our canonical MLBAM IDs so we can join with projections and valuations. Two-way players like Ohtani have two Yahoo IDs (one batting, one pitching) that both map to one MLBAM player — this requires a dedicated mapping table rather than a single column on Player.

### Steps

1. Create `src/fantasy_baseball_manager/domain/yahoo_player.py`:
   - `YahooPlayerMap` frozen dataclass: `yahoo_player_key`, `player_id` (our internal ID), `player_type` ("batter" | "pitcher"), `yahoo_name`, `yahoo_team`, `yahoo_positions`.
   - The `player_type` field handles two-way players: Ohtani has two rows — one with `player_type="batter"`, one with `player_type="pitcher"` — both pointing to the same `player_id`.
2. Add migration `016_yahoo_player_map.sql` for the mapping table with UNIQUE constraint on `(yahoo_player_key)`.
3. Create `src/fantasy_baseball_manager/yahoo/player_map.py`:
   - `YahooPlayerMapper` class that resolves Yahoo players to internal IDs.
   - Resolution strategy (ordered): exact Yahoo key lookup in mapping table → MLBAM ID from Yahoo player metadata (Yahoo provides MLB player IDs in their API) → name + team fuzzy matching (reuse existing `search_by_name` from `PlayerRepo`).
   - For unresolved players, log a warning and allow manual mapping via `fbm yahoo map-player <yahoo-key> <player-name>`.
   - Auto-detect two-way players: if Yahoo returns the same underlying MLB ID for two different Yahoo player keys, create both mapping rows pointing to the same internal player_id.
4. Create `src/fantasy_baseball_manager/domain/roster.py`:
   - `Roster` frozen dataclass: `team_key`, `league_key`, `season`, `week`, `as_of` (date), `entries` (tuple of `RosterEntry`).
   - `RosterEntry` frozen dataclass: `player_id`, `yahoo_player_key`, `player_name`, `position`, `roster_status` ("active" | "bench" | "il" | "na"), `acquisition_type` ("draft" | "add" | "trade").
5. Add migration `017_yahoo_roster.sql` for roster snapshots.
6. Create `src/fantasy_baseball_manager/yahoo/roster_source.py` implementing `DataSource`:
   - Fetches rosters for all teams in a league.
   - Maps Yahoo player keys to internal IDs via `YahooPlayerMapper`.
   - Returns roster entries as dicts for the `Loader` pipeline.
7. Add `fbm yahoo rosters --league <name>` CLI command that displays all teams' rosters with player names, positions, and roster status. Highlight your team.
8. Add `fbm yahoo my-roster --league <name>` CLI command showing your roster with projected stats and valuations joined in.
9. Write tests: player mapping resolution (exact, MLBAM fallback, name fuzzy match), two-way player handling, roster fetch/parse, unresolved player warnings.

### Acceptance criteria

- Yahoo player IDs are mapped to internal MLBAM-based player IDs.
- Two-way players (e.g., Ohtani) have two Yahoo ID rows mapping to one internal player.
- Unresolved players are logged with enough context for manual resolution.
- `fbm yahoo rosters` shows all teams' rosters.
- `fbm yahoo my-roster` shows your roster with projections and valuations.
- Roster data is persisted for historical snapshots.

## Phase 3: Draft results and live draft tracking

Fetch historical draft results from Yahoo and wire live draft polling into the existing draft tracker plan.

### Context

The live draft tracker roadmap (docs/plans/live-draft-tracker.md) defines a `DraftEngine` with manual pick entry via a REPL. With Yahoo integration, picks can be ingested automatically via API polling during a live draft — the user watches picks flow in rather than typing them manually. Historical draft results are also needed for keeper evaluation (Phase 4).

Yahoo's draft results endpoint returns all picks with player keys, team keys, round, pick number, and cost (for auction). During a live draft, polling this endpoint at an interval reveals new picks.

### Steps

1. Create `src/fantasy_baseball_manager/domain/draft_pick.py`:
   - `DraftPick` frozen dataclass: `league_key`, `season`, `round`, `pick`, `team_key`, `yahoo_player_key`, `player_id`, `player_name`, `position`, `cost` (for auction, None for snake).
   - This aligns with the `DraftPick` type in the live draft tracker plan but adds Yahoo-specific fields.
2. Add migration `018_yahoo_draft.sql` for draft picks with UNIQUE constraint on `(league_key, season, round, pick)`.
3. Create `src/fantasy_baseball_manager/yahoo/draft_source.py` implementing `DataSource`:
   - Fetches draft results for a league/season via `get_draft_results(league_key)`.
   - Maps player keys to internal IDs via `YahooPlayerMapper`.
   - Handles both snake and auction formats.
4. Add `fbm yahoo draft-results --league <name> --season <year>` CLI command to fetch and display historical draft results.
5. Create `src/fantasy_baseball_manager/yahoo/draft_poller.py`:
   - `DraftPoller` class that polls Yahoo's draft results endpoint on a configurable interval (default 5 seconds).
   - Compares current results against last-known state to detect new picks.
   - Emits new picks as they arrive (callback-based: `on_pick(DraftPick)`).
   - Handles connection interruptions gracefully (retry with backoff).
6. Wire the poller into the `DraftEngine` from the live draft tracker plan:
   - `DraftEngine.ingest_yahoo_pick(draft_pick)` — records a pick from the poller (same as manual `pick()` but sourced from Yahoo).
   - The REPL still works for manual corrections (`undo`, `pick` override) alongside auto-ingested picks.
7. Add `fbm yahoo draft-live --league <name>` CLI command that starts the live draft session:
   - Initializes `DraftEngine` with valuations and league settings.
   - Starts `DraftPoller` in a background thread.
   - Enters the interactive REPL (from draft tracker Phase 3) with auto-pick ingestion.
   - Displays recommendations after each pick (from draft tracker Phase 2).
8. Support fetching draft results for past seasons (needed for keeper eval). Yahoo provides access to prior seasons via game key: `{game_key_for_year}.l.{league_id}`.
9. Write tests: draft result parsing (snake and auction), poller state diff detection, DraftEngine integration with poller, historical multi-season fetch.

### Acceptance criteria

- Historical draft results are fetched and stored for any accessible season.
- Live draft poller detects new picks within the polling interval.
- Picks auto-populate in the DraftEngine without manual entry.
- Manual corrections (undo, override) still work alongside auto-ingestion.
- Recommendations display after each auto-ingested pick.
- Both snake and auction draft formats are supported.
- Past seasons' draft results are accessible for keeper analysis.

## Phase 4: League history and keeper evaluation

Walk the historical seasons of keeper leagues to build the context needed for keeper decisions — who was kept at what cost, how keeper values have trended, and what the optimal keeper set is for the upcoming season.

### Context

The keeper surplus value roadmap (docs/plans/keeper-surplus-value.md) defines `KeeperCost` storage, surplus calculation, and adjusted draft pool logic. That plan assumes costs are imported from CSV. With Yahoo integration, we can derive keeper costs directly from draft history: a player's keeper cost is typically their prior-year auction price (or draft round), and keeper selections are visible as players who appear on a team's roster before the draft.

This phase bridges Yahoo API data into the keeper surplus value plan, automating what would otherwise be manual CSV imports.

### Steps

1. Create `src/fantasy_baseball_manager/yahoo/history_source.py`:
   - `YahooLeagueHistorySource` that walks accessible past seasons for a league.
   - Yahoo reuses the same league ID across seasons but uses different game keys per year. Discover available seasons via the games collection endpoint filtered by `game_codes=mlb`.
   - For each accessible season: fetch draft results, final rosters, and transactions.
2. Create `src/fantasy_baseball_manager/services/keeper_cost_derivation.py`:
   - `derive_keeper_costs(draft_picks, current_rosters, league_rules)` → list of `KeeperCost`.
   - Logic: for each player on a team's current roster who was also on that team at end of prior season, infer keeper status. Cost = prior year's draft price (auction) or draft round equivalent (snake). Apply league-specific inflation rules if configured.
   - Handle players acquired via trade (keeper cost follows the player, not the team).
   - Handle free agent pickups (typically kept at minimum cost or league-defined floor).
3. Add `fbm yahoo keeper-costs --league <name> --season <year>` CLI command that:
   - Fetches league history if not already cached.
   - Derives keeper costs from draft + roster history.
   - Displays each player's keeper cost, how it was derived (drafted, traded, FA pickup), and years kept.
   - Allows manual overrides via `fbm yahoo keeper-cost-set <player> --cost <n>` for league-specific rules the auto-derivation can't infer.
4. Wire derived keeper costs into the keeper surplus value plan's `compute_surplus()`:
   - `fbm yahoo keeper-decisions --league <name> --season <year> --system <valuation-system>` — shows all keepable players ranked by surplus value, using Yahoo-derived costs.
   - Supports `--threshold <n>` for minimum surplus to recommend keeping.
5. Add keeper history tracking:
   - `KeeperHistory` frozen dataclass: `player_id`, `league_key`, `seasons_kept` (list of years), `cost_history` (list of costs by year).
   - CLI command `fbm yahoo keeper-history --league <name> --player <name>` showing a player's keeper trajectory.
6. Write tests: season discovery, cost derivation from draft data (auction + snake), trade-acquired player cost tracking, FA pickup cost floor, multi-year keeper history assembly.

### Acceptance criteria

- Accessible historical seasons are discovered and fetched.
- Keeper costs are correctly derived from auction draft prices.
- Trade-acquired players retain their original draft cost.
- Free agent pickups use the configured cost floor.
- `fbm yahoo keeper-decisions` shows surplus-ranked keeper recommendations.
- Manual cost overrides take precedence over auto-derived costs.
- Keeper history shows multi-year cost trajectories.

## Phase 5: Transaction log and league activity

Ingest transactions (adds, drops, trades, waiver claims) to maintain an up-to-date view of league activity and support in-season features.

### Context

With auth, rosters, draft data, and keeper evaluation in place, the final foundational piece is ongoing transaction tracking. This enables in-season features: waiver wire analysis (who was just dropped?), trade monitoring, and roster change history. It also keeps our roster snapshots current without requiring full re-syncs.

### Steps

1. Create `src/fantasy_baseball_manager/domain/transaction.py`:
   - `Transaction` frozen dataclass: `transaction_key`, `league_key`, `type` ("add" | "drop" | "trade" | "waiver"), `timestamp`, `status` ("successful" | "vetoed" | "pending").
   - `TransactionPlayer` frozen dataclass: `transaction_key`, `player_id`, `yahoo_player_key`, `source_team_key`, `dest_team_key`, `type` ("add" | "drop").
2. Add migration `019_yahoo_transaction.sql`.
3. Create `src/fantasy_baseball_manager/yahoo/transaction_source.py` implementing `DataSource`:
   - Fetches league transactions, maps player keys, categorizes by type.
   - Supports incremental fetch (only transactions after a given timestamp) to avoid re-processing.
4. Add `fbm yahoo transactions --league <name> --days <n>` CLI command showing recent league activity.
5. Add `fbm yahoo refresh --league <name>` command that incrementally syncs rosters, transactions, and standings. This is the "bring everything up to date" command.
6. Write tests: transaction parsing by type, incremental fetch, roster update from transactions.

### Acceptance criteria

- Transactions are fetched, categorized, and stored.
- Incremental sync avoids re-processing old transactions.
- `fbm yahoo refresh` brings all league data up to date.
- Adds/drops/trades are correctly attributed to teams.

## Ordering

**Phase 1 → Phase 2 → Phase 3 → Phase 4** is the critical path for draft prep and keeper decisions.

- **Phase 1** (OAuth + client) is a prerequisite for everything. No Yahoo data without auth.
- **Phase 2** (player mapping + rosters) is required by all downstream phases. Player ID crosswalk is the bridge between Yahoo and our projection system.
- **Phase 3** (draft results + live draft) depends on Phase 2 for player mapping. Historical draft results are also needed by Phase 4.
- **Phase 4** (keeper evaluation) depends on Phases 2 and 3. It also depends on the keeper surplus value roadmap's Phase 1-2 (domain types + surplus calculation) being implemented — those phases should be done first or in parallel.
- **Phase 5** (transactions) is independent of Phases 3-4 and can be done at any point after Phase 2. It's lower priority for draft prep but becomes important once the season starts.

**Cross-roadmap dependencies:**
- Phase 3 builds on the **live draft tracker roadmap** (Phases 1-3: state engine, recommendations, REPL). Those phases should be implemented first or concurrently.
- Phase 4 builds on the **keeper surplus value roadmap** (Phases 1-2: cost storage, surplus calculation). Phase 1 of that roadmap can be simplified since Yahoo provides the cost data that the CSV import was designed for.
- Downstream features not in this roadmap (sit/start, trade finder, waiver wire, category balance) all depend on Phase 2 (rosters + player mapping) as their foundation.
