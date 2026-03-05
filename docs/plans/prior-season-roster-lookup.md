# Prior-Season Roster Lookup Roadmap

The keeper-decisions flow currently fails when the prior season hasn't been explicitly synced. The user must run `fbm yahoo sync --season 2025` before `fbm yahoo keeper-decisions` will work, because the system looks up `team_repo.get_user_team(prior_league_key)` which requires teams to already exist in the local database.

The sibling project (`fantasy-baseball`) solves this by walking the Yahoo renewal chain and fetching rosters directly from the prior season's league via the Yahoo API — no prior sync required. This roadmap brings that approach to fbm: auto-syncing prior-season metadata and fetching the end-of-season roster on demand during keeper cost derivation.

## Status

| Phase | Status |
|-------|--------|
| 1 — Auto-sync prior season | done (2026-03-04) |
| 2 — End-of-season roster fetch | in progress |
| 3 — League-wide keeper intelligence | not started |

## Phase 1: Auto-sync prior season

When keeper-costs or keeper-decisions needs prior-season data and it isn't in the DB, automatically sync the prior season's league metadata (teams) so the user team can be resolved.

### Context

Today, `derive_and_store_keeper_costs` and `derive_best_n_keeper_costs` both call `team_repo.get_user_team(prior_league_key)`. If the prior season was never synced, this returns `None` and raises `ValueError("No user team found…")`. The user must manually run `fbm yahoo sync --season <prior>` first — a confusing prerequisite that isn't obvious.

The `sync_league_metadata` service already knows how to fetch and store league + team data from Yahoo. The fix is to call it automatically for the prior-season league key when teams are missing.

### Steps

1. Extract the "ensure prior season is synced" logic into a helper function (e.g., `ensure_prior_season_synced`) in `services/yahoo_keeper.py` or a new `services/prior_season.py` module.
2. The helper should: check `team_repo.get_user_team(prior_league_key)`. If `None`, call `sync_league_metadata` for the prior league key (using the game key extracted from `prior_league_key`), then commit. Re-check and raise if still `None`.
3. Call this helper from `yahoo_keeper_costs` and `yahoo_keeper_decisions` CLI commands, right after `_resolve_prior_league_key`, before calling the derive functions.
4. Add unit tests: (a) when prior teams exist, no sync happens; (b) when missing, sync is called and teams are populated; (c) when sync still yields no user team, a clear error is raised.

### Acceptance criteria

- `fbm yahoo keeper-decisions --league keeper` succeeds without a prior `fbm yahoo sync --season 2025`.
- Prior-season league and team metadata is stored in the DB after the auto-sync.
- If the prior-season league genuinely has no matching user team (e.g., user didn't play that year), a clear error message is shown.
- Existing tests continue to pass; no regressions in the sync or keeper flows.

## Phase 2: End-of-season roster fetch

Fetch the final roster from the prior season rather than the week-1 roster, so keeper candidates reflect end-of-season rosters (including trade acquisitions, waiver pickups, etc.).

### Context

`YahooRosterSource.fetch_team_roster` calls `client.get_roster(team_key)` which returns the team's **current** roster. For a completed prior season, Yahoo returns the final roster — but the code currently hardcodes `week=1` in the call from `derive_and_store_keeper_costs`. More importantly, when a league's season is over, the Yahoo `team/{team_key}/roster/players` endpoint returns the end-of-season roster by default, which is correct. However, the system doesn't validate or log which week's roster it received.

The `fantasy-baseball` project explicitly navigates to the prior season's league object and fetches rosters from there, making the intent clear. This phase aligns fbm's approach: use the prior-season team key (which Yahoo resolves to the final roster) and add logging/validation to confirm the roster is from the expected season.

### Steps

1. Add an optional `date` or `week` parameter to `YahooFantasyClient.get_roster` so callers can explicitly request a specific week's roster (Yahoo supports `?week=N` on the roster endpoint). Default behavior (no param) returns the final/current roster.
2. Update `derive_and_store_keeper_costs` and `derive_best_n_keeper_costs` to drop the hardcoded `week=1` and instead pass no week (to get the final roster for completed seasons).
3. Add logging in `YahooRosterSource.fetch_team_roster` that records how many roster entries were fetched and the team key used, to aid debugging.
4. Add integration-style tests with fixture data: given a prior-season team key, the fetch returns the full end-of-season roster (not a partial/week-1 roster).

### Acceptance criteria

- Keeper cost derivation uses the end-of-season roster for the prior season (waiver pickups and trade acquisitions appear as keeper candidates).
- `get_roster` supports an optional week parameter for cases where a specific week is needed.
- Debug logging shows roster source details (team key, entry count).
- Tests verify roster parsing with realistic prior-season fixture data.

## Phase 3: League-wide keeper intelligence

Fetch all teams' prior-season rosters (not just the user's) so the keeper optimizer can account for other teams' likely keepers when computing replacement levels.

### Context

The `fantasy-baseball` project fetches rosters for **all teams** in the prior-season league and uses this to estimate which players other teams will keep. This lets the surplus calculator exclude other teams' probable keepers from the draft pool, producing more accurate replacement levels.

Currently fbm's `keeper optimize` and `keeper decisions` commands only know about the user's roster. Other teams' keepers are not considered, which inflates the apparent draft pool and understates replacement levels.

### Steps

1. Add a `fetch_all_team_rosters` method to `YahooRosterSource` (the existing `fetch_all_rosters` is a stub). It should iterate all teams from `team_repo` for the prior league key, calling `fetch_team_roster` for each.
2. Create an `estimate_other_keepers` function that takes all teams' rosters, applies a simple heuristic (e.g., top N players by projected value on each team are likely keepers), and returns a set of player IDs.
3. Wire the estimated other-keeper set into `compute_surplus` / `keeper optimize` as an optional parameter. When provided, these players are excluded from the replacement pool.
4. Add a `--estimate-league-keepers` flag to `yahoo keeper-decisions` and `yahoo keeper-costs` that enables this behavior.
5. Add tests for the estimation heuristic and its effect on surplus calculations.

### Acceptance criteria

- `--estimate-league-keepers` flag fetches all teams' rosters from the prior season.
- Other teams' probable keepers are excluded from the replacement pool.
- Surplus values shift when league-wide keeper estimation is enabled vs. disabled.
- The heuristic is configurable (e.g., max keepers per team from league config).
- Tests cover: estimation logic, surplus adjustment, and edge cases (teams with fewer players than max keepers).

## Ordering

**Phase 1 → Phase 2 → Phase 3** (strictly sequential).

Phase 1 is the critical fix — it removes the manual sync prerequisite and unblocks the keeper-decisions workflow. Phase 2 improves accuracy by ensuring the correct roster snapshot is used. Phase 3 adds strategic depth by considering the full league context.

No external roadmap dependencies. This builds on the completed Yahoo Integration Improvements roadmap.
