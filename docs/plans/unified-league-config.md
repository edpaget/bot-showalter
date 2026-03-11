# Unified League Config Roadmap

Currently Yahoo integration and league settings use two separate config namespaces with independent names: `[leagues.h2h]` defines league settings (format, categories, positions) while `[yahoo.leagues.keeper]` defines Yahoo API config (league_id, keeper format). This means keeper costs get stored under the Yahoo league name (e.g., `"keeper"`) but the web server and CLI look them up using the league settings name (e.g., `"h2h"`), causing "No scenarios available" in the keeper planner.

This roadmap merges Yahoo league config into the league settings as `[leagues.*.yahoo]` sub-tables, so each league has a single canonical name used everywhere — config, database, CLI, and web UI. The top-level `[yahoo]` section retains only auth credentials and `default_league`.

## Status

| Phase | Status |
|-------|--------|
| 1 — Merge config parsing | done (2026-03-10) |
| 2 — Migrate CLI and services | done (2026-03-11) |
| 3 — Database migration and cleanup | in progress |

## Phase 1: Merge config parsing

Move Yahoo league config from `[yahoo.leagues.*]` into `[leagues.*.yahoo]` sub-tables. Update `config_yahoo.py` to discover Yahoo leagues by scanning `[leagues.*.yahoo]` instead of `[yahoo.leagues.*]`. Update `config_league.py` to ignore the `yahoo` key when parsing league settings.

### Context

`config_yahoo.py` reads `[yahoo.leagues]` into `YahooConfig.leagues: dict[str, YahooLeagueConfig]`. `config_league.py` reads `[leagues]` into `LeagueSettings`. These two dicts are keyed by different names for the same league. There's no link between them, so code that needs both (e.g., web.py building the keeper planner) can't match them up.

### Steps

1. Update `config_league.py`'s `parse_league()` to skip the `yahoo` key when parsing — it should not raise on unknown fields, just ignore the sub-table.
2. Update `config_yahoo.py`'s `load_yahoo_config()` to scan `data["leagues"]` for entries with a `yahoo` sub-table, building `YahooLeagueConfig` from `[leagues.<name>.yahoo]` instead of `[yahoo.leagues.<name>]`. Fall back to `[yahoo.leagues.*]` if present (backward compat for this phase).
3. Update `YahooConfig` to remove its `leagues` dict — replace with a function `load_yahoo_league(name: str, config_dir: Path) -> YahooLeagueConfig | None` that reads `[leagues.<name>.yahoo]`.
4. Update `fbm.toml`: move `[yahoo.leagues.keeper]` fields into `[leagues.h2h.yahoo]` (renaming the league to use the canonical settings name). Remove `[yahoo.leagues]` section.
5. Update tests in `test_config_yahoo.py` and `test_league_config.py` to use new TOML structure.
6. Keep `[yahoo]` top-level section for `client_id`, `client_secret`, `default_league`.

### Acceptance criteria

- `load_yahoo_config()` reads Yahoo league config from `[leagues.*.yahoo]` sub-tables.
- `load_league()` successfully parses leagues with a `[yahoo]` sub-table, ignoring it.
- `resolve_default_league()` still resolves the default league name.
- All existing config tests pass with updated TOML fixtures.
- `fbm.toml` uses the new structure with no `[yahoo.leagues]` section.

## Phase 2: Migrate CLI and services

Update all CLI commands and services to use the unified league name. Commands that previously accepted Yahoo league names now accept league settings names (which have Yahoo config embedded).

### Context

Yahoo CLI commands (e.g., `fbm yahoo sync --league keeper`) accept Yahoo league names from `[yahoo.leagues]`. Keeper CLI commands accept league settings names from `[leagues]`. The web command's `--league` flag is a league settings name. After phase 1, these are all the same namespace, so the `--league` flag everywhere refers to `[leagues.*]` keys.

### Steps

1. Update `yahoo.py` CLI commands: change `--league` help text and resolution to use `load_league()` + check for `.yahoo` sub-config. Replace `config.leagues[league]` lookups with the new `load_yahoo_league()` function.
2. Update `derive_and_store_keeper_costs()` and `derive_best_n_keeper_costs()` callers to pass the canonical league name (from `LeagueSettings.name`).
3. Update `web.py` keeper cost lookup — it now uses `league_name` directly since the league settings name matches the keeper cost storage name.
4. Update `draft.py` `--exclude-keepers` to resolve via league settings name.
5. Update CLI help text across all commands to reference `[leagues]` consistently.
6. Update integration tests that wire up Yahoo + keeper flows.

### Acceptance criteria

- `fbm yahoo sync --league h2h` works (reads Yahoo config from `[leagues.h2h.yahoo]`).
- `fbm yahoo keeper-costs --league h2h` stores costs with `league="h2h"`.
- `fbm keeper decisions --league h2h` finds costs stored by Yahoo commands.
- `fbm web --league h2h` builds the keeper planner with costs from `league="h2h"`.
- No CLI command references `[yahoo.leagues]` in help text.

## Phase 3: Database migration and cleanup

Migrate existing `keeper_cost` and `league_keeper` rows from old Yahoo league names to canonical league settings names. Remove backward-compat fallback from phase 1.

### Context

Existing databases have keeper costs stored under Yahoo league names (e.g., `league="keeper"`). After phases 1-2, all new writes use the canonical name (e.g., `"h2h"`). Old data needs migrating so the keeper planner can find it.

### Steps

1. Add a SQL migration that updates `keeper_cost.league` and `league_keeper.league` columns. Since the mapping is user-specific (depends on their `fbm.toml`), provide a CLI command `fbm keeper migrate-league --from keeper --to h2h` rather than a hardcoded migration.
2. Remove the backward-compat fallback in `config_yahoo.py` that reads `[yahoo.leagues.*]`.
3. Update any remaining test fixtures that use old-style Yahoo league names.
4. Add a deprecation warning if `[yahoo.leagues]` is present in `fbm.toml`, directing users to move config to `[leagues.*.yahoo]`.

### Acceptance criteria

- `fbm keeper migrate-league --from keeper --to h2h` updates all `keeper_cost` and `league_keeper` rows.
- Loading a `fbm.toml` with `[yahoo.leagues]` prints a deprecation warning.
- No code reads from `[yahoo.leagues]` except the deprecation check.
- Keeper planner finds costs after migration.

## Ordering

Phases are sequential — each builds on the previous:

1. **Phase 1** (config parsing) must land first since it defines the new TOML structure.
2. **Phase 2** (CLI/services) depends on phase 1 for the new config loading functions.
3. **Phase 3** (data migration) should come last since it removes backward compat and migrates existing data.

Phase 1 is the highest priority — it unblocks the keeper planner for users who set up `[leagues.*.yahoo]`.
