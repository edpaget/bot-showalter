# Refactoring Roadmap

This document tracks identified technical debt and refactoring opportunities in the fantasy baseball codebase.

## High Priority

### 1. Replace Global Factory Pattern with Dependency Container

**Status:** ✅ Completed (commit `bdf64a6`)

**Problem:** Seven CLI modules used identical global factory anti-pattern for dependency injection.

**Solution implemented:** Created `ServiceContainer` class in `src/services/container.py` that:
- Provides lazy initialization of dependencies via `@cached_property`
- Supports explicit injection for testing via constructor parameters
- Manages: `data_source`, `id_mapper`, `roster_source`, `blender`, `yahoo_league`
- Uses `get_container()` / `set_container()` for global access with reset capability

All CLI modules now use the container pattern. Tests use `set_container(ServiceContainer(...))` for clean dependency injection.

---

### 2. Consolidate Cache Wrapper Classes

**Status:** ✅ Completed

**Problem:** `cache/sources.py` contained three nearly identical classes with duplicated cache logic.

**Solution implemented:** Extracted shared cache-or-fetch logic into a `_cached_fetch()` helper function that:
- Handles cache hit/miss, serialization, deserialization, and logging uniformly
- Takes serializer functions as parameters
- Wrapper classes now delegate to this helper with type-specific serializers

**Result:** Reduced from 182 lines to 158 lines (13% reduction). More importantly, the cache logic is now in one place (DRY), making it easier to modify caching behavior consistently.

---

### 3. Type-Safe Metadata in Pipeline

**Status:** ✅ Completed

**Problem:** `PlayerRates.metadata` was typed as `dict[str, object]`, requiring unsafe `cast()` calls.

**Solution implemented:** Created `PlayerMetadata` TypedDict in `src/pipeline/types.py` with all known metadata fields:
- Input metadata (pa_per_year, ip_per_year, is_starter, position, team, etc.)
- Platoon split metadata (rates_vs_lhp, rates_vs_rhp, pct_vs_lhp, pct_vs_rhp)
- Enhanced playing time metadata (injury_factor, age_pt_factor, volatility_factor, base_pa, base_ip)
- Pitcher normalization metadata (observed_babip, expected_babip, expected_lob_pct)
- Statcast adjuster metadata (statcast_blended, statcast_xwoba, pitcher_xera, etc.)
- GB residual adjuster metadata (gb_residual_adjustments)

**Result:** Removed cast() calls and type: ignore comments from pipeline stages. IDE autocompletion and type checking now work for metadata access.

---

## Medium Priority

### 4. Split Large CLI Modules

**Status:** ✅ Partially completed

**Problem:** Several CLI modules exceed 400 lines and mix multiple concerns:

| File | Lines (before) | Lines (after) | Concerns Mixed |
|------|----------------|---------------|----------------|
| `keeper/cli.py` | 512 | 428 | Data orchestration, display formatting, validation, optimization |
| `draft/cli.py` | 477 | 364 | Projections, simulation, display, shared utilities |
| `agent/tools.py` | 392 | — | Many similar tool definitions |

**Specific issues:**
- ~~`draft/cli.py` contains `build_projections_and_positions()` which is imported by `keeper/cli.py` — cross-module dependency~~ ✅ Fixed
- ~~Duplicated `_cli_context()` context manager in all 3 CLI modules~~ ✅ Fixed
- Display logic (table formatting) mixed with data orchestration
- Validation logic scattered throughout

**Solution implemented (Phase 1):**
- Extracted `cli_context()` to `src/services/cli.py` — removed ~75 lines of duplication across 3 modules
- Extracted `build_projections_and_positions()` to `src/shared/orchestration.py` — resolved cross-module dependency
- All CLI modules now import shared utilities instead of defining them locally

**Remaining work:**
- ~~Extract display/formatting to separate modules (item #9)~~ ✅ Done via rich tables refactoring
- Consider subcommand modules for complex CLIs

---

### 5. Builder Creates Internal Dependencies

**Status:** ✅ Completed

**Problem:** `pipeline/builder.py` created all dependencies internally via `create_cache_store()` calls in 6 places, making it hard to inject mocks for testing or swap cache implementations.

**Solution implemented:**
- Added `cache_store` parameter to `PipelineBuilder.__init__()` for constructor injection
- Added `with_cache_store()` builder method for fluent API injection
- Added `_get_cache_store()` helper that returns injected store or creates one lazily
- All 6 internal `create_cache_store()` calls now use `self._get_cache_store()`

**Result:** Tests can now inject fake cache stores to verify behavior without hitting real caches. Cache implementation can be swapped by callers.

---

### 6. Extract Common CLI Setup Pattern

**Status:** ✅ Completed

**Problem:** Every CLI module duplicated setup boilerplate and had near-identical helper functions like `_get_id_mapper()`, `_get_roster_source()`, etc.

**Solution implemented:**
- Added `app_config`, `cache_store`, `cache_key` properties to `ServiceContainer`
- Added `invalidate_caches()` method to `ServiceContainer`
- Added `roster_league` property for keeper CLI needs
- Removed duplicate `_get_id_mapper()`, `_get_roster_source()`, `_get_yahoo_league()`, `_invalidate_caches()` from all CLI modules
- CLI modules now use `get_container()` properties directly: `.id_mapper`, `.roster_source`, `.data_source`, `.yahoo_league`, `.cache_store`, `.cache_key`

**Result:** Removed ~150 lines of duplicated helper functions across 3 CLI modules. Single source of truth for service creation in `ServiceContainer`.

---

### 7. Config Uses Global Override State

**Status:** ✅ Completed

**Problem:** `config.py` used mutable global state for CLI overrides.

**Solution implemented:**
- Added `league_id` and `season` parameters to `ServiceConfig` and `create_config()`
- Created `_cli_context()` context manager in each CLI module that:
  - Sets up a `ServiceContainer` with overrides
  - Automatically resets the container on exit
  - Respects existing containers (for test injection)
- CLI modules (`league/cli.py`, `draft/cli.py`, `keeper/cli.py`) now use context managers instead of `apply_cli_overrides()`/`clear_cli_overrides()`
- Legacy global override mechanism (`set_cli_overrides`, `clear_cli_overrides`) preserved for backwards compatibility

**Result:** CLI operations are now isolated and don't leak state. Tests can inject fake containers that won't be overwritten.

---

## Lower Priority

### 8. Extend Rich Table Formatting to Remaining CLIs

**Status:** ✅ Completed

**Problem:** While draft, keeper, and league CLIs now use Rich tables, several other CLI modules still use manual string formatting with fixed-width columns.

**Solution implemented:**
- **marcel/cli.py** — Converted `format_batting_table()` → `print_batting_table()`, `format_pitching_table()` → `print_pitching_table()` using rich tables
- **valuation/cli.py** — Converted `_format_value_table()` → `_print_value_table()` using rich tables
- **draft/simulation_report.py** — Converted `format_pick_log()` → `print_pick_log()`, `format_team_roster()` → `print_team_roster()`, `format_standings()` → `print_standings()` using rich tables
- **evaluation/cli.py** — Converted `_format_stat_accuracy_table()` → `_print_stat_accuracy_table()`, `_format_rank_accuracy()` → `_print_rank_accuracy()`, etc. using rich tables
- **ml/cli.py** — Converted `list_cmd()` and `info_cmd()` to use rich tables for model listing and feature importance display

**Result:** All CLI modules now use consistent rich table formatting for tabular output.

---

### 9. Resolve `type: ignore` Comments

**Status:** ✅ Mostly completed

**Problem:** 14 files contained `type: ignore` comments.

**Solution implemented:**
- **draft/results.py** — Added proper `yahoo_fantasy_api.League` type annotation via TYPE_CHECKING import
- **draft/positions.py** — Used `cast(Iterable[object], ...)` for player dict iteration
- **draft/cli.py** — Added `cast("yahoo_fantasy_api.League", ...)` for yahoo_league access
- **keeper/cli.py** — Added `cast("yahoo_fantasy_api.League", ...)` for roster_league access
- **config.py** — Used `cast()` for dict type narrowing in `apply_cli_overrides()` and `load_league_settings()`
- **pipeline/stages/split_data_source.py** — Changed `fetch: object` to `Callable[[int], list[BattingSeasonStats]]`
- **pipeline/stages/pitcher_normalization.py** — Replaced `isinstance(x, dict)` with `x is not None` for proper TypedDict narrowing
- **pipeline/types.py** — Removed now-unnecessary type: ignore on TypedDict default
- **evaluation/harness.py** — Used `_bucket_players_pitching()` instead of reusing batting function with casts

**Remaining (external library issues):**
- `agent/core.py`, `agent/cli.py` — langchain/langgraph incomplete type stubs (3 comments)
- `pipeline/park_factors.py` — pandas `iterrows()` not recognized by `ty` (1 comment)

**Result:** Reduced from 14 files to 3 files with type: ignore comments. Remaining comments are due to third-party library limitations.

---

### 10. Column Formatting Duplication

**Status:** ✅ Completed (commits `67e62db`, `34dd6fd`)

**Problem:** Three CLI modules define similar column specification patterns:

- `league/cli.py` (lines 30-79)
- `draft/cli.py`
- `keeper/cli.py`

Each defines column specs with lambdas for data extraction and formatting.

**Solution implemented:**
- Replaced manual f-string table formatting with `rich.table.Table` in all CLI modules
- `draft/cli.py` and `keeper/cli.py` now use inline rich table definitions without duplicated column spec patterns
- `league/cli.py` retains domain-specific column specs (stat categories) that aren't duplicated elsewhere

**Result:** Column formatting duplication eliminated. Each CLI module uses rich tables with context-appropriate column definitions.

---

### 11. Missing Contract Tests for Protocol Implementations

**Status:** Not started

**Problem:** Multiple implementations of protocols (e.g., `RateComputer`, `RateAdjuster`, `PlayingTimeProjector`) are tested independently but no contract tests verify they all satisfy the protocol correctly.

**Solution:** Add parametrized contract tests that run against all implementations of each protocol.

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total source files | 107 |
| Total test files | 101 |
| Largest file | `keeper/cli.py` (428 lines) |
| Global state locations | 1 (services/container.py) |
| Files with `type: ignore` | 3 (reduced from 14) |
| Duplicated cache wrappers | ✅ Consolidated |
| Column formatting duplication | ✅ All CLI modules now use rich tables |
| CLI modules needing split | 1 (`agent/tools.py`) |

## Completed Items

1. ✅ **Replace Global Factory Pattern with Dependency Container** — `ServiceContainer` now manages all CLI dependencies centrally
2. ✅ **Consolidate Cache Wrapper Classes** — Extracted `_cached_fetch()` helper, reducing duplication in cache/sources.py
3. ✅ **Type-Safe Metadata in Pipeline** — Created `PlayerMetadata` TypedDict with all known fields, eliminating cast() calls
4. ✅ **Config Uses Global Override State** — CLI modules now use `cli_context()` context managers with `ServiceConfig` overrides instead of mutable global state
5. ✅ **Extract Common CLI Setup Pattern** — Removed duplicate helper functions from CLI modules; all services now accessed via `ServiceContainer` properties
6. ✅ **Split Large CLI Modules (Phase 1)** — Extracted `cli_context()` to `services/cli.py` and `build_projections_and_positions()` to `shared/orchestration.py`
7. ✅ **Builder Creates Internal Dependencies** — Added `cache_store` injection to `PipelineBuilder` via constructor and `with_cache_store()` method
8. ✅ **Column Formatting Duplication** — Replaced f-string tables with rich tables across all CLI modules, eliminating duplicated column spec patterns
9. ✅ **Resolve type: ignore Comments** — Fixed type annotations in 9 files; remaining 3 files have external library stub issues
10. ✅ **Extend Rich Table Formatting to Remaining CLIs** — Converted marcel, valuation, draft/simulation_report, evaluation, and ml CLIs to use rich tables
