# CLI Consistency Roadmap

The CLI has grown organically and accumulated inconsistencies in how common parameters — `--system`, `--version`, and `--season` — are defaulted and documented across 60+ commands. Version defaults silently split between `"1.0"` (draft, report, valuations, web) and `"production"` (keeper, yahoo), causing commands to use wrong valuations when the user doesn't explicitly pass `--version`. Season defaults are hardcoded to `2026` in 18+ locations, requiring a manual sweep every year. There is no single source of truth for these defaults.

This roadmap centralizes defaults into `fbm.toml` (overrideable via `fbm.local.toml`), replaces all hardcoded season values with dynamic computation, and standardizes parameter naming and help text across every command group.

## Status

| Phase | Status |
|-------|--------|
| 1 — Config-driven defaults infrastructure | done (2026-03-08) |
| 2 — Dynamic season computation | not started |
| 3 — Migrate all commands to config-driven defaults | not started |
| 4 — Standardize parameter naming and help text | not started |

## Phase 1: Config-driven defaults infrastructure

Add `system` and `version` keys to `[common]` in `fbm.toml` and create a shared module that CLI commands can use to resolve defaults.

### Context

Today, non-model CLI commands (draft, keeper, yahoo, report, etc.) never read `fbm.toml`. Their `--system` and `--version` defaults are hardcoded in each function signature — `"zar"` and `"1.0"` in draft/report/valuations/web, `"zar"` and `"production"` in keeper/yahoo. The `"1.0"` default is a bug: production valuations use version `"production"`, so draft/report commands silently find no valuations unless the user explicitly passes `--version production`. The existing `load_toml()` + `deep_merge()` infrastructure already supports `fbm.local.toml` overrides, but only `load_config()` (for model training) uses it.

### Steps

1. Add default keys to `fbm.toml` under `[common]`:
   ```toml
   [common]
   system = "zar"
   version = "production"
   ```
2. Create `src/fantasy_baseball_manager/cli/_defaults.py` with a `CliDefaults` frozen dataclass holding `system`, `version`, and `season` (phase 2 adds dynamic season). Provide a `load_cli_defaults(config_dir: Path | None = None) -> CliDefaults` function that reads `[common]` from `load_toml()` and falls back to hardcoded values if keys are missing.
3. Move `current_season()` from `agent/prompt.py` to a shared location (`domain/season.py` or similar) so both the agent and `_defaults.py` can import it. Re-export from `agent/prompt.py` to avoid breaking existing imports.
4. Add tests for `load_cli_defaults` covering: base defaults from toml, local override via `fbm.local.toml`, and fallback when keys are missing.

### Acceptance criteria

- `fbm.toml` has `system` and `version` under `[common]` with values `"zar"` and `"production"`.
- `load_cli_defaults()` returns a `CliDefaults` with system, version, and season populated from TOML (with `fbm.local.toml` override support).
- `current_season()` is importable from a shared module outside `agent/`.
- Existing code continues to work unchanged (no commands migrated yet).

## Phase 2: Dynamic season computation

Replace all hardcoded `= 2026` season defaults with the dynamic `current_season()` function, and fix the `report.py` hardcoded `current_year = 2026` bug.

### Context

18+ CLI commands default `--season` to `2026` via hardcoded values in function signatures. The `report injury-profile` command has a worse variant: `current_year = 2026` is used internally (not even a parameter) to compute the lookback range. The `current_season()` function already exists in `agent/prompt.py` with the correct logic (returns current year Jan–Sep, next year Oct–Dec), but only `chat` and `discord` use it. Typer doesn't support calling a function for a default value in `Annotated[int, typer.Option(...)]`, so the approach is to use `None` as the parameter default and resolve to `current_season()` at the top of the function body.

### Steps

1. In every command that has `season: ... = 2026`, change the default to `None` and resolve at runtime:
   ```python
   season: Annotated[int | None, typer.Option("--season", help="Season year")] = None,
   ```
   Then at the top of the function body:
   ```python
   if season is None:
       season = current_season()
   ```
2. Fix `report.py` `injury-profile` command: replace `current_year = 2026` with `current_season()`.
3. Fix `models.ensemble.params.season = 2026` in `fbm.toml` — this is a model param not a CLI default, but add a comment noting it must be updated seasonally (or consider reading from `current_season()` in the ensemble model).
4. Update affected tests that rely on the hardcoded `2026` default.

### Acceptance criteria

- No CLI command has a hardcoded season default (grep for `= 2026` in `cli/commands/` returns zero hits).
- `report injury-profile` uses `current_season()` instead of `current_year = 2026`.
- All commands that accept `--season` default to `current_season()` when not provided.
- Tests pass without hardcoded year assumptions.

## Phase 3: Migrate all commands to config-driven system/version defaults

Replace every hardcoded `--system` and `--version` default with values from `CliDefaults`.

### Context

After phase 1 provides the infrastructure and phase 2 handles season, the remaining work is mechanical: update each command's `--system` and `--version` parameters to use `None` defaults and resolve from `CliDefaults`. This affects ~40 commands across 8 files. The pattern is the same everywhere: change the default to `None`, call `load_cli_defaults()` early in the function, and use the resolved values. Commands that don't accept `--system` or `--version` (model training, ingest, etc.) are unaffected.

### Steps

1. Update `draft.py` (~17 commands): change `system = "zar"` and `version = "1.0"` defaults to `None`, resolve from `CliDefaults`.
2. Update `mock_draft.py` (3 commands): same pattern.
3. Update `report.py` (~5 commands with `--system`/`--version`): same pattern.
4. Update `valuations.py` (~2 commands): same pattern.
5. Update `keeper.py` (~8 commands): change `version = "production"` to `None`, resolve from `CliDefaults`. Also fix `data_dir = "data"` (missing `./` prefix) to use the same `_DataDirOpt` pattern as other files.
6. Update `yahoo.py` (~6 commands with `--system`/`--version`): same pattern.
7. Update `web.py` (1 command): same pattern.
8. Update `standalone.py` (`compare` command with hardcoded `"steamer"` / `"zips"`): these are projection system names for comparison, not valuation defaults — leave as-is but verify.
9. Consider whether `load_cli_defaults()` should be called once per command invocation or cached. Since `load_toml()` reads files, a module-level cache (via `functools.lru_cache`) may be appropriate.
10. Update or add tests verifying that when `--system`/`--version` are omitted, the TOML-configured defaults are used.

### Acceptance criteria

- No command has a hardcoded `"1.0"` version default (the bug that started this effort).
- No command has a hardcoded `"zar"` system default — all resolve from `[common]` in `fbm.toml`.
- `keeper.py` `data_dir` default matches the `"./data"` convention used everywhere else.
- A user can override the default system/version in `fbm.local.toml` and have it take effect across all commands without passing CLI flags.
- All existing tests pass; new tests verify config-driven resolution.

## Phase 4: Standardize parameter naming and help text

Normalize `--system` and `--version` help text, option names, and conventions across all commands.

### Context

Minor inconsistencies remain after phases 1-3:
- Help text varies: `"Valuation system"`, `"Valuation system name"`, `"Filter by valuation system"`, `"Original valuation system"`, `"Projection system"`.
- `keeper.py` doesn't use `--data-dir` flag name (just `help="Data directory"` without explicit option name).
- Some commands use `_DataDirOpt` type alias, others inline the `Annotated[...]` expression.
- `--version` help text is just `"Valuation version"` everywhere (consistent, but could note the default source).

### Steps

1. Standardize `--system` help text:
   - Valuation-consuming commands: `"Valuation system"` (draft, keeper, yahoo, report value commands, web, mock_draft).
   - Projection-consuming commands: `"Projection system"` (projections lookup, draft needs with projection context).
   - Model commands: keep `"System/version (e.g. statcast-gbm/latest)"` — these use the combined format and are a different concept.
2. Standardize `--version` help text to `"Valuation version"` everywhere (already mostly consistent).
3. Ensure all `data_dir` parameters use the `_DataDirOpt` type alias with `--data-dir` flag name. Move `_DataDirOpt` to `cli/_defaults.py` so it's defined once and imported everywhere, instead of redefined in each command file.
4. Standardize `--season` help text to `"Season year"` everywhere.
5. Verify `--league` defaults are consistent within each command group (draft commands should all use the same default; keeper commands should all use the same default).

### Acceptance criteria

- `--system` help text is `"Valuation system"` for all valuation-consuming commands and `"Projection system"` for projection-consuming commands.
- `_DataDirOpt` is defined once in `cli/_defaults.py` and imported by all command files.
- All `data_dir` parameters use `"./data"` as the default (no bare `"data"`).
- `--season` help text is `"Season year"` everywhere.
- No command file redefines `_DataDirOpt` locally.

## Ordering

Phases are sequential:
- **Phase 1** is foundational — creates the config infrastructure that phases 2-3 depend on.
- **Phase 2** can technically be done independently of phase 1 (it only needs `current_season()` relocation), but is cleaner after phase 1 since `CliDefaults` can hold the computed season.
- **Phase 3** depends on phase 1 (needs `CliDefaults` and `load_cli_defaults()`).
- **Phase 4** is cosmetic cleanup and can land any time after phase 3, or even incrementally alongside it.

This roadmap supersedes the season-related aspects of the `valuation-version-consistency` roadmap (phase 1 of that roadmap already landed `--version` on keeper/yahoo commands; phase 2 pushes filtering into the repo layer and remains independent of this work).
