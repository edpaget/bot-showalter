# Yahoo League Web Integration Roadmap

Expose the associated Yahoo league in the web UI so that when the server starts with a Yahoo league configured, the UI is league-aware. Today all Yahoo interaction happens through CLI commands — the web UI has no way to see league metadata, team rosters, standings, or trigger keeper cost derivation from Yahoo data. The only Yahoo touchpoint in the web API is the live draft poller, which requires a league key the UI can't discover.

This roadmap threads Yahoo league context through AppContext and adds GraphQL queries and frontend components so the UI can display league info, browse team rosters, show standings, and feed Yahoo-derived keeper costs into the keeper planner — all without leaving the browser.

## Status

| Phase | Status |
|-------|--------|
| 1 — League context in AppContext and WebConfig | done (2026-03-11) |
| 2 — League metadata and standings queries | not started |
| 3 — Team rosters query and viewer | not started |
| 4 — Yahoo-aware keeper planner | not started |

## Phase 1: League context in AppContext and WebConfig

Thread the Yahoo league identity through the web server so the frontend can discover whether a Yahoo league is configured, its name, league key, season, and keeper settings.

### Context

Today `AppContext` has a `yahoo_poller_manager` but no league metadata. The frontend's `webConfig` query only returns projection/valuation system-version pairs. When the server starts with `--yahoo-config-dir`, it loads `YahooConfig` and creates the poller manager, but discards all league metadata. The frontend has no way to know a Yahoo league exists, what its key is, or whether it's a keeper league.

### Steps

1. Add a `YahooLeagueInfo` frozen dataclass (or extend `WebConfig`) with fields: `league_key`, `league_name`, `season`, `num_teams`, `is_keeper`, `max_keepers`, `user_team_name`. This is a simple read-only snapshot, not a live API object.
2. In `web.py`, when `--yahoo-config-dir` is provided, look up the Yahoo league and user team from the database repos (`yahoo_league_repo.get_by_league_key`, `yahoo_team_repo.get_user_team`) and populate the info object. Store it on `AppContext` (new optional field).
3. Add a `YahooLeagueInfoType` GraphQL type and expose it as an optional field on `WebConfigType` (or as a top-level `yahooLeague` query). Return `null` when no Yahoo league is configured.
4. Run `bun run codegen` to generate the frontend types. Add the new field to the existing `WEB_CONFIG_QUERY` (or add a new query).
5. Create a small `LeagueBadge` component that renders in the nav bar or header when `yahooLeague` is non-null, showing the league name and season. This gives immediate visual confirmation that the server is league-connected.

### Acceptance criteria

- `webConfig` (or a new `yahooLeague` query) returns league metadata when the server is started with a Yahoo league configured; returns null otherwise.
- The frontend displays the league name somewhere visible when connected.
- No behavioral changes when the server is started without Yahoo config.

## Phase 2: League metadata and standings queries

Add GraphQL queries to browse teams and standings from the Yahoo league data already synced to the database.

### Context

The CLI commands `yahoo standings` and `yahoo sync` populate `yahoo_team` and `yahoo_team_season_stats` tables. This data is already in the database but not exposed via GraphQL. Showing standings and team lists in the UI gives context for keeper decisions and draft strategy.

### Steps

1. Add `YahooTeamType` and `YahooStandingsEntryType` GraphQL types in `web/types.py`, mapping from the existing `YahooTeam` and `YahooTeamSeasonStats` domain models.
2. Add `yahoo_teams(league_key)` and `yahoo_standings(league_key, season)` queries to the GraphQL schema. Use the existing repos (`yahoo_team_repo`, `yahoo_team_stats_repo`) — no new data fetching needed.
3. Wire the repos into `AppContext` or make them accessible via `container`. The `yahoo_team_repo` is already constructed in `web.py` for the poller; extend its availability.
4. Add a `LeagueView` frontend page (new route `/league`) with a standings table and team list. Link it from the nav bar (only visible when Yahoo league is configured, using the phase 1 `yahooLeague` field).
5. Run codegen, add tests for the new queries.

### Acceptance criteria

- `yahooTeams` query returns all teams in the league with name, manager, and user-team flag.
- `yahooStandings` query returns team stats/rankings for the league season.
- A new `/league` page displays standings and team list.
- The `/league` nav link is hidden when no Yahoo league is configured.

## Phase 3: Team rosters query and viewer

Expose team roster snapshots so the user can browse what each team has — essential for keeper analysis and understanding the competitive landscape.

### Context

`yahoo rosters` CLI command syncs roster snapshots into `yahoo_roster_snapshot` + `yahoo_roster_entry` tables. Each snapshot has a team, date, and list of players with positions and acquisition types. The `yahoo_roster_repo` can retrieve latest rosters by team or league. This data is valuable for keeper planning (seeing what other teams might keep) but currently requires the CLI to view.

### Steps

1. Add `YahooRosterType` and `YahooRosterEntryType` GraphQL types mapping from `Roster` and `RosterEntry` domain models. Include the player's internal `player_id` (from `yahoo_player_map`) so the frontend can link to player profiles.
2. Add `yahoo_rosters(league_key)` query that returns latest roster for each team, and `yahoo_roster(team_key, league_key)` for a single team. Use `yahoo_roster_repo.get_by_league_latest()`.
3. Wire `yahoo_roster_repo` into the web server (add to `AppContext` or make accessible).
4. Add a roster viewer to the `/league` page — click a team in the standings to expand/show their roster. Link player names to the existing player drawer.
5. Run codegen, add tests.

### Acceptance criteria

- `yahooRosters` query returns latest roster snapshot for every team in the league.
- Each roster entry includes player name, position, and internal player ID (when mapped).
- The `/league` page lets users click a team to view their roster.
- Player names in rosters open the player drawer.

## Phase 4: Yahoo-aware keeper planner

Connect the keeper planner page to Yahoo league data so the user can derive keeper costs from Yahoo and see other teams' likely keepers, all from the web UI.

### Context

Today keeper costs must be populated via CLI (`yahoo keeper-costs`) before starting the server. The keeper planner page takes season and max_keepers but has no way to trigger cost derivation or see other teams' keeper situations. The building blocks exist: `derive_and_store_keeper_costs()` in `services/yahoo_keeper.py`, `estimate_other_keepers()` in `services/keeper_league_analysis.py`, and the `LeagueKeeper` model. This phase wires them into the web API.

### Steps

1. Add a `deriveKeeperCosts(leagueKey, season)` GraphQL mutation that calls the existing `derive_and_store_keeper_costs()` service and rebuilds the `KeeperPlannerService` instance on `AppContext`. This replaces the CLI `yahoo keeper-costs` step. Requires the Yahoo API client on AppContext (already available when poller is configured).
2. Add a `yahooKeeperOverview(leagueKey, season)` query that returns each team's roster with estimated surplus values — reusing the logic from the CLI `yahoo keeper-league` command. This lets the user see what other teams are likely to keep.
3. Update the `KeeperPlannerView` frontend: when a Yahoo league is configured, show a "Sync Keeper Costs from Yahoo" button that calls the new mutation. Auto-populate `maxKeepers` from the league config.
4. Add an "Other Teams' Keepers" section to the keeper planner page showing estimated keepers per team from the `yahooKeeperOverview` query.
5. Run codegen, add tests.

### Acceptance criteria

- A mutation can derive keeper costs from Yahoo data without leaving the web UI.
- The keeper planner auto-populates max_keepers from Yahoo league config when available.
- The keeper planner shows estimated keepers for other teams.
- The keeper planner works identically to today when no Yahoo league is configured (no regressions).

## Ordering

Phases are sequential — each builds on the previous:

- **Phase 1** is the foundation: without league context in AppContext, subsequent phases have no way to know the league key or whether Yahoo is configured.
- **Phase 2** depends on phase 1 for the league key and the conditional nav link.
- **Phase 3** depends on phase 2 for the `/league` page and team list to attach rosters to.
- **Phase 4** depends on phases 1-3 for league context, and requires the Yahoo API client on AppContext (already partially wired in phase 1).

Phase 1 is small and high-value — it unblocks the rest and immediately tells the user the server is Yahoo-connected. Phase 4 is the most complex but also the highest payoff for keeper league users.
