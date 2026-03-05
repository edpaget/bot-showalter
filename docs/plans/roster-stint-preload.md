# Roster Stint Preload Roadmap

The Discord bot (and CLI agent) fails to look up players by team for 2026 because `roster_stint` only contains Lahman data through 2025 — and Lahman won't have 2026 data until the season ends. The `MlbApiPlayerTeamProvider` has a silent `season - 1` fallback and swallows API errors, which means failures present as "no data" rather than surfacing an actionable error.

This roadmap adds a CLI command to pre-populate `roster_stint` from the MLB API (so team lookups work offline without a runtime API call) and removes the silent fallbacks so missing data produces clear errors.

## Status

| Phase | Status |
|-------|--------|
| 1 — Pre-populate roster stints from MLB API | done (2026-03-04) |
| 2 — Remove silent fallbacks | in progress |

## Phase 1: Pre-populate roster stints from MLB API

Add a new ingest command that fetches current team assignments from the MLB Stats API and writes them to `roster_stint`, so the bot doesn't need a live API call at runtime.

### Context

The existing `ingest roster` command reads from Lahman's `Appearances.csv`, which only covers completed seasons. For the current/upcoming season (2026), team assignments are only available via the MLB Stats API. Currently the `MlbApiPlayerTeamProvider` calls the API at runtime, but this is fragile (network issues on the bot host cause silent failures) and means every `find_players` call for the current season hits the API.

### Steps

1. Add a new CLI command `fbm ingest roster-api --season <year>` that:
   - Calls `fetch_mlb_active_teams(season)` to get `mlbam_id -> abbreviation` mappings.
   - Resolves each `mlbam_id` to an internal `player_id` via the player repo.
   - Resolves each team abbreviation to a `team_id` via the team repo (auto-upsert teams if needed, matching what `ingest roster` does).
   - Upserts rows into `roster_stint` with `season`, `start_date` set to today (or a `--as-of` parameter), and `end_date` null.
2. Handle players not yet in the `player` table — log/skip them (they won't have projections anyway).
3. Handle re-runs gracefully — upsert on the unique constraint so running the command again updates team assignments (e.g., after a trade).
4. Add tests using a fake fetcher (no real API calls in tests).

### Acceptance criteria

- `fbm ingest roster-api --season 2026` populates `roster_stint` with ~1000+ rows for 2026.
- `find_players(season=2026, team="NYY")` returns results using only `roster_stint` data (no API call needed at query time).
- Re-running the command updates changed assignments without duplicating rows.
- Tests cover the happy path, unknown-player skip, and team resolution.

## Phase 2: Remove silent fallbacks

Remove the `season - 1` fallback and the silent `except Exception` in `MlbApiPlayerTeamProvider` so missing data produces clear errors instead of wrong or empty results.

### Context

`MlbApiPlayerTeamProvider.get_player_teams()` has two silent fallbacks: (1) if no roster stints exist for the requested season, it quietly uses `season - 1`; (2) if the MLB API call fails, it catches all exceptions and logs a warning. Both mask data gaps — the bot returns stale team assignments or empty results rather than telling the user something is wrong. After phase 1, roster stints will always be pre-populated for active seasons, so these fallbacks are no longer needed.

### Steps

1. Remove the `season - 1` fallback in `MlbApiPlayerTeamProvider.get_player_teams()` — if `roster_stint` has no rows for the requested season, return the empty dict (don't silently use the previous year).
2. Remove the `try/except` around the MLB API overlay — let API errors propagate. If the fetcher is `None` (no API configured), that's fine; but if it's wired up and fails, the caller should know.
3. Update `PlayerBiographyService.find()` so that when both roster stints and the provider return empty for a team query, the "no results" message is explicit about the season having no roster data (vs. the team simply not existing).
4. Update existing tests that rely on the fallback behavior.
5. Update the agent system prompt to instruct the LLM to tell users to run the ingest command if roster data is missing, rather than guessing.

### Acceptance criteria

- Querying `get_player_teams()` for a season with no roster stints returns an empty dict (does not fall back to previous season).
- An MLB API failure raises an exception instead of being silently swallowed.
- The `find_players` tool returns a clear message like "No roster data for season 2026. Run `fbm ingest roster-api --season 2026` to load it." when roster stints are missing.
- No existing tests break (tests that relied on fallback behavior are updated to either pre-populate data or expect the new error).

## Ordering

Phase 1 must land first — it provides the mechanism to pre-populate data. Phase 2 removes the fallbacks, which would break things if there were no way to populate the data in the first place. Both phases are small and can land in quick succession.
