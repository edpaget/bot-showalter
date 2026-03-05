# Player Biography Fuzzy Team Lookup Roadmap

The `PlayerBiographyService.find()` method currently requires an exact team abbreviation (e.g. `"NYY"`, `"LAD"`) with a Lahman↔modern alias fallback. Users must know the precise abbreviation — searching for `"Yankees"`, `"New York Yankees"`, `"NY"`, or `"dodgers"` all fail silently. This roadmap adds fuzzy team-name resolution so that the `team` filter accepts full names, partial names, city names, and common nicknames, and surfaces helpful suggestions when no match is found.

## Status

| Phase | Status |
|-------|--------|
| 1 — Team name resolver | done (2026-03-04) |
| 2 — Wire into biography service | done (2026-03-05) |
| 3 — CLI integration & error messages | done (2026-03-05) |

## Phase 1: Team name resolver

Build a standalone `TeamResolver` that converts free-text team input into a canonical team abbreviation.

### Context

The `team_aliases.py` module maps Lahman↔modern abbreviations but has no concept of full team names, city names, or nicknames. The `team` table in the database stores `abbreviation`, `name`, `league`, and `division`, but `find()` only queries by abbreviation. There's no fuzzy matching for team strings anywhere in the codebase (unlike player names, which have `difflib.get_close_matches` in `player_resolver.py`).

### Steps

1. Create `src/fantasy_baseball_manager/team_resolver.py` with a `TeamResolver` class that accepts a `TeamRepo` dependency.
2. On first call, build a lookup dictionary mapping lowercase variants to abbreviation: full name (`"new york yankees"` → `"NYY"`), city (`"new york"` → `["NYY", "NYM"]`), nickname (`"yankees"` → `"NYY"`), abbreviation (`"nyy"` → `"NYY"`), and Lahman aliases.
3. Implement a `resolve(query: str) -> list[str]` method with a tiered strategy:
   - **Exact match** — direct hit in the lookup dict (case-insensitive).
   - **Substring match** — query is a substring of a known name/city/nickname.
   - **Fuzzy match** — `difflib.get_close_matches` against all known names/cities/nicknames (cutoff ~0.6).
4. Return a list of matching abbreviations (single element for unambiguous, multiple for ambiguous like `"New York"`).
5. Add a `TeamResolverProtocol` to `repos/protocols.py` so it can be injected.
6. Write tests covering: exact abbreviation, full name, city-only, nickname-only, case insensitivity, Lahman abbreviations, ambiguous city (returns multiple), misspelled name (fuzzy), and no-match.

### Acceptance criteria

- `resolve("Yankees")` → `["NYY"]`
- `resolve("NYY")` → `["NYY"]`
- `resolve("KCA")` → `["KC"]`
- `resolve("New York")` returns both `"NYY"` and `"NYM"` (ambiguous)
- `resolve("dodgers")` → `["LAD"]` (case-insensitive)
- `resolve("Yankeez")` → `["NYY"]` (fuzzy)
- `resolve("xyzabc")` → `[]` (no match)
- All tests pass with `uv run pytest`.

## Phase 2: Wire into biography service

Update `PlayerBiographyService.find()` to use `TeamResolver` so the `team` filter accepts free-text input.

### Context

Currently `find(team="NYY")` works but `find(team="Yankees")` silently returns an empty list. After phase 1 provides fuzzy resolution, the service needs to use it and handle ambiguity gracefully.

### Steps

1. Add `TeamResolver` as an optional dependency to `PlayerBiographyService.__init__()`.
2. In `find()`, when `team` is provided and a `TeamResolver` is available, resolve the team string first. If resolution returns exactly one abbreviation, use it. If multiple, query all matching teams and merge results.
3. Keep the existing direct-abbreviation path as a fast-path (try exact abbreviation + Lahman alias before hitting the resolver).
4. When resolution returns zero results, raise a descriptive `ValueError` with the unresolved query.
5. Update existing tests to verify backward compatibility (exact abbreviations still work).
6. Add new tests for fuzzy team input flowing through `find()`.

### Acceptance criteria

- `find(team="Yankees", season=2024)` returns the same results as `find(team="NYY", season=2024)`.
- `find(team="New York", season=2024)` returns players from both NYY and NYM.
- `find(team="NYY", season=2024)` still works (backward compatible).
- `find(team="xyzabc", season=2024)` raises `ValueError`.
- All existing `test_player_biography.py` tests still pass.

## Phase 3: CLI integration & error messages

Surface team resolution in the CLI `bio` command and provide user-friendly feedback.

### Context

The `bio` CLI currently passes the `--team` flag value straight through to `find()`. With fuzzy resolution, the CLI should echo which team(s) it resolved to, and when the query is ambiguous, show the user the candidates so they can refine.

### Steps

1. Update the `bio` CLI command handler to catch ambiguous results (multiple teams) and display them as suggestions (e.g., `"'New York' matches multiple teams: NYY (New York Yankees), NYM (New York Mets). Use a more specific name or abbreviation."`).
2. When resolution fails, show a helpful error message listing similar team names if fuzzy matching found near-misses.
3. When resolution succeeds unambiguously, print a brief note (e.g., `"→ Resolved 'Yankees' to NYY"`) so the user sees what was matched.
4. Add CLI-level tests for the new output messages.

### Acceptance criteria

- `fbm bio --team Yankees --season 2024` prints resolution note and player results.
- `fbm bio --team "New York" --season 2024` prints disambiguation message listing NYY and NYM.
- `fbm bio --team xyzabc --season 2024` prints a clear error, not a traceback.
- All CLI tests pass.

## Ordering

Phases are strictly sequential: phase 2 depends on the resolver from phase 1, and phase 3 depends on the service changes from phase 2. No external roadmap dependencies.
