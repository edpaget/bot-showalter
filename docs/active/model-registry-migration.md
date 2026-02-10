# Model Registry Migration

## Overview

Unify the four separate model persistence modules (`ModelStore`, `MTLModelStore`, `MLEModelStore`, `ContextualModelStore`) behind a shared `registry` package. The registry extracts ~90% duplicated persistence logic into `BaseModelStore`, adds version tracking, and provides a `ModelRegistry` facade for cross-type operations. The infrastructure layer is complete; what remains is wiring the registry into CLI commands and adopting versioning in training workflows.

---

## What's Done

| Component | Status | Notes |
|-----------|--------|-------|
| `registry/serializers.py` | Complete | `JoblibSerializer`, `TorchParamsSerializer`, `ModelSerializer` protocol |
| `registry/base_store.py` | Complete | `BaseModelStore`, `ModelMetadata` with JSON sidecar, legacy format support |
| `registry/mtl_store.py` | Complete | `MTLBaseModelStore` with batter/pitcher class dispatch |
| `registry/registry.py` | Complete | `ModelRegistry` facade: `list_all`, `next_version`, `versions_of`, `compare` |
| `registry/factory.py` | Complete | `create_gb_store`, `create_mtl_store`, `create_mle_store`, `create_model_registry` |
| Legacy store delegation | Complete | `ModelStore`, `MTLModelStore`, `MLEModelStore` delegate to `BaseModelStore` internally |
| `ServiceContainer` | Complete | Exposes `model_registry` as a `@cached_property` |
| `PipelineBuilder` | Partial | Accepts `model_registry`, passes `model_dir` to legacy store constructors |
| CLI `ml train` wired to registry | Complete | `train_cmd` sources `ModelStore` from `create_model_registry()` |
| CLI `ml list/delete/info` wired | Complete | All 3 commands source `ModelStore` from registry |
| CLI `ml train-mtl` wired to registry | Complete | `train_mtl_cmd` sources `MTLModelStore` from registry |
| Test coverage | Complete | 55+ tests across serializers, base store, registry, backward compat |

---

## Remaining Steps

### ~~Step 1: Wire CLI `ml train` to the Registry~~ (Complete)

All 5 ML CLI commands (`train`, `list`, `delete`, `info`, `train-mtl`) now obtain their model stores from `create_model_registry()` via a shared `_get_registry()` helper in `cli.py`.

---

### ~~Step 2: Wire CLI `ml train-mtl` to the Registry~~ (Complete)

Completed as part of Step 1 — `train_mtl_cmd` sources `MTLModelStore(model_dir=registry.mtl_store.model_dir)`.

---

### ~~Step 3: Wire CLI `contextual pretrain` and `finetune` to the Registry~~ (Complete)

Both `pretrain_cmd` and `finetune_cmd` now obtain `ContextualModelStore` from `create_model_registry()` via a shared `_get_registry()` helper in `contextual/cli.py`, matching the pattern used in `ml/cli.py`.

---

### ~~Step 4: Add `--version` Flag to Training Commands~~ (Complete)

Both `ml train` and `ml train-mtl` now accept an optional `--version` flag. When omitted, `resolve_version()` auto-increments by computing `max(next_version(batter), next_version(pitcher))` so both player types share a consistent version number. Version 1 uses the bare name (`"default"`), version 2+ uses `"{name}_v{version}"` (`"default_v2"`).

---

### ~~Step 5: Update `ml list` to Use Cross-Type Listing~~ (Complete)

`list_cmd` now uses `registry.list_all(model_type=model_type)` instead of a GB-only `ModelStore`. The table includes Name, Model Type, Player Type, Version, Training Years, Stats, and Created columns. An optional `--type` / `-t` flag filters by model type (e.g., `ml list --type mtl`).

---

### ~~Step 6: Add `ml compare` Command~~ (Complete)

Added `ml compare` CLI command that uses `registry.compare()`. The command outputs side-by-side metadata (name, version, training years, created), training years diff, and a metrics table with deltas. Errors clearly if either model doesn't exist. 4 tests in `tests/ml/test_cli_compare.py`.

---

### ~~Step 7: Have PipelineBuilder Pass Stores Directly~~ (Complete)

Pipeline stages (`GBResidualAdjuster`, `MTLBlender`, `MTLRateComputer`) now accept `BaseModelStore`/`MTLBaseModelStore` directly instead of legacy wrappers. `PipelineBuilder` passes `registry.gb_store` and `registry.mtl_store` directly. Each stage has a `_default_*_store()` factory for standalone use. `GBResidualAdjuster._ensure_models_loaded` now uses `load_params` + `ResidualModelSet.from_params`. MTL stages use `load_batter`/`load_pitcher` instead of `load_batter_model`/`load_pitcher_model`. Tests updated accordingly.

---

### Step 8: Drop Legacy Wrapper Classes

Once all callers use the registry or base stores directly, the legacy wrapper classes become unnecessary. This is the final cleanup step and should only be done after all other steps are verified.

**Files to simplify:**
- `src/fantasy_baseball_manager/ml/persistence.py` — `ModelStore` can be replaced by `BaseModelStore` with `JoblibSerializer`
- `src/fantasy_baseball_manager/ml/mtl/persistence.py` — `MTLModelStore` can be replaced by `MTLBaseModelStore`
- `src/fantasy_baseball_manager/minors/persistence.py` — `MLEModelStore` can be replaced by `BaseModelStore` with `JoblibSerializer`

**Acceptance criteria:**
- No remaining imports of legacy store classes outside of tests
- All existing tests updated to use registry types
- Full test suite passes

---

## Implementation Order and Risk

| Step | Effort | Risk | Dependency |
|------|--------|------|------------|
| ~~1. Wire `ml train`~~ | ~~Small~~ | ~~Low~~ | **Done** |
| ~~2. Wire `ml train-mtl`~~ | ~~Small~~ | ~~Low~~ | **Done** |
| ~~3. Wire contextual CLI~~ | ~~Small~~ | ~~Low~~ | **Done** |
| ~~4. Add `--version` flag~~ | ~~Medium~~ | ~~Low~~ | **Done** |
| ~~5. Update `ml list`~~ | ~~Medium~~ | ~~Low~~ | **Done** |
| ~~6. Add `ml compare`~~ | ~~Medium~~ | ~~None (new feature)~~ | **Done** |
| ~~7. Direct store injection~~ | ~~Medium~~ | ~~Medium (touches pipeline stages)~~ | **Done** |
| 8. Drop legacy wrappers | Large | Medium (breaking internal API) | Steps 1-7 |

Steps 1-3 can be done in parallel. Steps 4-6 can be done in parallel after 1-2. Step 7 is independent but benefits from 1-3. Step 8 is the final cleanup and should be done last.

---

## Verification

After each step:
1. `uv run pytest` — all existing tests pass
2. `uv run pytest tests/registry/` — registry tests pass
3. `uv run ruff check src tests` — no lint issues
4. `uv run ty check src tests` — no type errors

After all steps:
5. Manual: `ml train --name test_v1` saves through registry
6. Manual: `ml list` shows models across all types
7. Manual: `ml compare test_v1 test_v2 --type gb_residual --player-type batter` shows metric diff
