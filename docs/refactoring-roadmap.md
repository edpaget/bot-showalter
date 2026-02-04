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

**Status:** Not started

**Problem:** Several CLI modules exceed 400 lines and mix multiple concerns:

| File | Lines | Concerns Mixed |
|------|-------|----------------|
| `keeper/cli.py` | 512 | Data orchestration, display formatting, validation, optimization |
| `draft/cli.py` | 477 | Projections, simulation, display, shared utilities |
| `agent/tools.py` | 392 | Many similar tool definitions |

**Specific issues:**
- `draft/cli.py` contains `build_projections_and_positions()` which is imported by `keeper/cli.py` — cross-module dependency
- Display logic (table formatting) mixed with data orchestration
- Validation logic scattered throughout

**Solution:**
- Extract shared data orchestration to `src/shared/orchestration.py`
- Extract display/formatting to separate modules
- Consider subcommand modules for complex CLIs

---

### 5. Builder Creates Internal Dependencies

**Status:** Not started

**Problem:** `pipeline/builder.py` (307 lines) creates all dependencies internally:

```python
def _build_adjusters(self) -> list[RateAdjuster]:
    if self._park_factors:
        adjusters.append(
            ParkFactorAdjuster(
                CachedParkFactorProvider(
                    delegate=FanGraphsParkFactorProvider(),
                    cache=create_cache_store()  # Creates dependency internally
                )
            )
        )
```

**Issues:**
- Hard to inject mocks for testing
- No way to swap cache implementations
- Tightly coupled to factory functions

**Solution:** Accept optional dependency overrides in constructor or add `.with_cache_store()` builder methods.

---

### 6. Extract Common CLI Setup Pattern

**Status:** Not started

**Problem:** Every CLI module duplicates setup boilerplate:

```python
config = create_config()
cache = create_cache_store(config)
cache_key = get_cache_key(config)
data_source = _data_source_factory()
```

**Solution:** Create a `CLIContext` dataclass bundling common dependencies.

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

### 8. Resolve `type: ignore` Comments

**Status:** Not started

**Problem:** 14 files contain `type: ignore` comments:

- `agent/core.py` — langchain/langgraph incomplete type stubs
- `draft/results.py`
- `pipeline/stages/adjusters.py`
- And others

**Solution:** Track and resolve as library type stubs improve; consider contributing stubs upstream.

---

### 9. Column Formatting Duplication

**Status:** Not started

**Problem:** Three CLI modules define similar column specification patterns:

- `league/cli.py` (lines 30-79)
- `draft/cli.py`
- `keeper/cli.py`

Each defines column specs with lambdas for data extraction and formatting.

**Solution:** Create shared table formatting utilities in `src/shared/tables.py`.

---

### 10. Missing Contract Tests for Protocol Implementations

**Status:** Not started

**Problem:** Multiple implementations of protocols (e.g., `RateComputer`, `RateAdjuster`, `PlayingTimeProjector`) are tested independently but no contract tests verify they all satisfy the protocol correctly.

**Solution:** Add parametrized contract tests that run against all implementations of each protocol.

---

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total source files | 105 |
| Total test files | 101 |
| Largest file | `keeper/cli.py` (512 lines) |
| Global state locations | 1 (services/container.py) |
| Files with `type: ignore` | 8 (reduced from 14) |
| Duplicated cache wrappers | ✅ Consolidated |
| CLI modules needing split | 3 |

## Completed Items

1. ✅ **Replace Global Factory Pattern with Dependency Container** — `ServiceContainer` now manages all CLI dependencies centrally
2. ✅ **Consolidate Cache Wrapper Classes** — Extracted `_cached_fetch()` helper, reducing duplication in cache/sources.py
3. ✅ **Type-Safe Metadata in Pipeline** — Created `PlayerMetadata` TypedDict with all known fields, eliminating cast() calls
4. ✅ **Config Uses Global Override State** — CLI modules now use `_cli_context()` context managers with `ServiceConfig` overrides instead of mutable global state
