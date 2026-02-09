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
| Test coverage | Complete | 55+ tests across serializers, base store, registry, backward compat |

---

## Remaining Steps

### Step 1: Wire CLI `ml train` to the Registry

**File:** `src/fantasy_baseball_manager/ml/cli.py` (line ~160)

Currently `ml train` creates `ModelStore()` directly. Update it to obtain the store from `ServiceContainer.model_registry`:

```python
# Before
model_store = ModelStore()

# After
from fantasy_baseball_manager.registry.factory import create_model_registry
registry = create_model_registry()
model_store = ModelStore(model_dir=registry.gb_store.model_dir)
```

Or, for full registry adoption:

```python
registry = create_model_registry()
registry.gb_store.save_params(model.get_params(), name, player_type, ...)
```

**Acceptance criteria:**
- `ml train --name default` saves through the registry path
- Existing saved models are still loadable (backward compat)
- Tests in `tests/ml/` continue to pass

---

### Step 2: Wire CLI `ml train-mtl` to the Registry

**File:** `src/fantasy_baseball_manager/ml/cli.py` (line ~617)

Same pattern as Step 1 but for `MTLModelStore`:

```python
# Before
model_store = MTLModelStore()

# After
registry = create_model_registry()
model_store = MTLModelStore(model_dir=registry.mtl_store.model_dir)
```

**Acceptance criteria:**
- `ml train-mtl` saves through the registry path
- Tests in `tests/ml/mtl/` continue to pass

---

### Step 3: Wire CLI `contextual pretrain` and `finetune` to the Registry

**File:** `src/fantasy_baseball_manager/contextual/cli.py` (lines ~519, ~800)

Contextual stays as a specialized store but should be obtained from the registry for consistent directory resolution:

```python
# Before
model_store = ContextualModelStore()

# After
registry = create_model_registry()
model_store = registry.contextual_store
```

**Acceptance criteria:**
- `contextual pretrain` and `contextual finetune` save through the registry
- Tests in `tests/contextual/` continue to pass

---

### Step 4: Add `--version` Flag to Training Commands

Add an optional `--version` flag to `ml train` and `ml train-mtl` that auto-increments using `registry.next_version()`:

```python
@app.command()
def train(
    name: str = "default",
    version: int | None = None,  # New flag
    ...
):
    registry = create_model_registry()
    if version is None:
        version = registry.next_version(name, "gb_residual", player_type)
    versioned_name = f"{name}_v{version}" if version > 1 else name
    ...
```

This enables saving multiple versions of a model (e.g., `default`, `default_v2`, `default_v3`) without overwriting previous ones.

**Acceptance criteria:**
- `ml train --name default` saves as `default` (version 1, backward compat)
- `ml train --name default --version 2` saves as `default_v2`
- Running `ml train --name default` when `default` already exists auto-increments to `default_v2`

---

### Step 5: Update `ml list` to Use Cross-Type Listing

**File:** `src/fantasy_baseball_manager/ml/cli.py` (line ~194)

Currently `ml list` only shows GB models. Update to use `registry.list_all()` for a unified view:

```python
# Before
store = ModelStore()
models = store.list_models()

# After
registry = create_model_registry()
models = registry.list_all()  # All model types
# Or filter: registry.list_all(model_type="gb_residual")
```

Update the display table to include a `Type` column showing `gb_residual`, `mtl`, `mle`, or `contextual`.

**Acceptance criteria:**
- `ml list` shows models from all stores
- Output includes model type, version, and training years
- Optional `--type` filter flag (e.g., `ml list --type mtl`)

---

### Step 6: Add `ml compare` Command

Add a new CLI command that uses `registry.compare()`:

```
ml compare default_v1 default_v2 --type gb_residual --player-type batter
```

This prints a side-by-side comparison of metrics between two model versions, useful for evaluating whether a retrained model is an improvement.

**Acceptance criteria:**
- Command outputs a table of metrics for both models
- Shows delta and percentage change for each metric
- Errors clearly if either model doesn't exist

---

### Step 7: Have PipelineBuilder Pass Stores Directly

**File:** `src/fantasy_baseball_manager/pipeline/builder.py` (lines ~522, ~551)

Currently `PipelineBuilder` extracts `model_dir` from the registry and creates legacy store instances. Update the pipeline stages (`GBResidualAdjuster`, `MTLBlender`) to accept `BaseModelStore` directly, eliminating the roundtrip through legacy wrappers:

```python
# Before (current)
gb_kwargs["model_store"] = ModelStore(model_dir=registry.gb_store.model_dir)

# After (direct injection)
gb_kwargs["model_store"] = registry.gb_store
```

This requires updating the type annotations in `GBResidualAdjuster` and `MTLBlender` to accept the base store types.

**Acceptance criteria:**
- Pipeline stages accept `BaseModelStore` / `MTLBaseModelStore`
- `PipelineBuilder` passes stores from registry directly
- Full pipeline integration test passes

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
| 1. Wire `ml train` | Small | Low | None |
| 2. Wire `ml train-mtl` | Small | Low | None |
| 3. Wire contextual CLI | Small | Low | None |
| 4. Add `--version` flag | Medium | Low | Steps 1-2 |
| 5. Update `ml list` | Medium | Low | None |
| 6. Add `ml compare` | Medium | None (new feature) | None |
| 7. Direct store injection | Medium | Medium (touches pipeline stages) | Steps 1-3 |
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
