# Test Performance Roadmap

Reduce test suite wall time and prevent future regressions by fixing slow test patterns, adding per-test timeout enforcement, and documenting test performance guidelines. The suite currently has 3,197 tests running in ~63s sequential. The three slowest files (`statcast_gbm/test_model.py`, `test_gbm_training.py`, `composite/test_model.py`) account for the vast majority of that time due to redundant inline ML training. Additionally, 99 tests are incorrectly marked `@pytest.mark.slow` despite doing no ML work, which inflates the `-m "not slow"` skip count and slows down the fast feedback loop.

## Status

| Phase | Status |
|-------|--------|
| 1 — Narrow `@pytest.mark.slow` scope | in progress |
| 2 — Share fixtures in `test_gbm_training.py` | not started |
| 3 — Share fixtures in `statcast_gbm/test_model.py` | not started |
| 4 — Add `pytest-timeout` and CLAUDE.md guidelines | not started |

## Phase 1: Narrow `@pytest.mark.slow` scope

Remove file-level `pytestmark = pytest.mark.slow` from both `test_gbm_training.py` and `statcast_gbm/test_model.py`. Apply `@pytest.mark.slow` only to classes that actually train ML models.

### Context

Both files use a file-level `pytestmark = pytest.mark.slow`, which marks every test in the file — including pure data/logic tests — as slow. This means `uv run pytest -m "not slow"` skips 99 tests that run instantly, making the fast feedback loop slower than it needs to be.

**`tests/models/test_gbm_training.py` — 55 tests incorrectly marked slow:**
- `TestTargetComparison`, `TestValidationResult`, `TestAblationResultValidation`, `TestCorrelationGroup`, `TestGroupedImportanceResult` — frozen dataclass checks
- `TestFindCorrelatedGroups` (7 tests) — pure correlation math, no ML
- `TestIdentifyPruneCandidates` (8 tests) — pure logic on pre-built results
- `TestExtractTargets` (11 tests), `TestExtractFeatures` (3 tests), `TestExtractSampleWeights` (3 tests) — data extraction
- `TestBuildCVFolds` (17 tests) — data manipulation, no ML

**`tests/models/statcast_gbm/test_model.py` — 44 tests incorrectly marked slow:**
- `TestStatcastGBMProtocol` (12 tests), `TestStatcastGBMPreseasonProtocol` (11 tests) — `isinstance()` checks
- `TestSampleWeightColumns` (3 tests), `TestSampleWeightTransformProperty` (2 tests), `TestSweepSupportedOperations` (2 tests), `TestMinActivityProperties` (4 tests), `TestResolveMinActivity` (4 tests) — property assertions
- `TestStatcastGBMPrepare` (1 test), `TestStatcastGBMEvaluate` (1 test), `TestStatcastGBMSeasonValidation` (4 tests) — no ML

### Steps

1. In `tests/models/test_gbm_training.py`: remove the file-level `pytestmark = pytest.mark.slow`. Add `@pytest.mark.slow` to only the 9 classes that do real ML training: `TestFitModels`, `TestScorePredictions`, `TestComputePermutationImportance`, `TestComputeGroupedPermutationImportance`, `TestEndToEndWithMissingTargets`, `TestGridSearchCV`, `TestValidatePruning`, `TestComputeCVPermutationImportance`, `TestSweepCV`.
2. In `tests/models/statcast_gbm/test_model.py`: remove the file-level `pytestmark = pytest.mark.slow`. Add `@pytest.mark.slow` to the classes that actually train/ablate/tune/sweep (all classes not listed as "incorrectly marked" above).
3. Run `uv run pytest -m "not slow" --co -q` and verify the previously-excluded 99 tests are now collected.
4. Run the full suite to confirm nothing broke.

### Acceptance criteria

- No file-level `pytestmark = pytest.mark.slow` in either file.
- `@pytest.mark.slow` is applied only to test classes that perform real ML training.
- `uv run pytest -m "not slow"` collects and runs at least 99 more tests than before this change.
- Full test suite still passes.

## Phase 2: Share fixtures in `test_gbm_training.py`

Convert inline helper functions to class-scoped fixtures so expensive ML operations run once per class instead of once per test.

### Context

`test_gbm_training.py` has zero pytest fixtures. It uses helper functions (`_make_cv_folds()`, `_make_validate_data()`, `_make_cv_importance_data()`, `_make_sweep_rows()`) called inline in every test method. This results in ~68 redundant ML operations. The file could eliminate at least 14 `fit_models()` / CV training cycles by sharing results.

### Steps

1. **`TestValidatePruning`** (8 tests, highest impact): `_make_validate_data()` calls `fit_models()` every time. Convert to a `@pytest.fixture(scope="class")` that returns the shared data + fitted models. Each test receives the pre-computed result and asserts on different properties. Saves 7 redundant `fit_models()` calls.
2. **`TestComputeCVPermutationImportance`** (8 tests): 5 of 8 tests call `compute_cv_permutation_importance()` with identical arguments. Create a class-scoped fixture that computes the result once and share it across those 5 tests. Tests with unique arguments (different `n_repeats`, `min_samples_leaf`) keep their own inline calls. Saves ~4 redundant CV training cycles.
3. **`TestGridSearchCV`** (11 tests): `_make_cv_folds()` itself is cheap (no ML), but each test then calls `grid_search_cv()` independently. Identify tests that use the same grid/params and create a class-scoped fixture for the shared `grid_search_cv()` result. Tests that need unique params keep their own calls.
4. **`TestComputePermutationImportance`** (6 tests): 4 tests use identical 4-row data. Create a class-scoped fixture for the shared `compute_permutation_importance()` result. Saves 3 redundant calls.
5. **`TestSweepCV`** (6 tests): Similar pattern — identify tests with identical sweep parameters and share the result via a class-scoped fixture.
6. Run the full test suite and confirm all tests pass.

### Acceptance criteria

- `TestValidatePruning` runs `fit_models()` at most once (via a class-scoped fixture).
- `TestComputeCVPermutationImportance` runs `compute_cv_permutation_importance()` at most 3 times (once per unique parameter set) instead of 8.
- No test behavior is changed — all assertions remain identical.
- Full test suite passes.
- `uv run pytest -o "addopts=" --durations=10 --durations-min=1.0 -q` shows improvement for tests in this file.

## Phase 3: Share fixtures in `statcast_gbm/test_model.py`

Consolidate redundant inline ML calls into shared class-scoped fixtures, following the pattern already used by `composite/test_model.py`.

### Context

Out of ~52 ML calls in this file, only 5 are in shared fixtures. The other ~47 are inline — each test constructs its own model and runs a full training pipeline. The biggest opportunities are:

- **Group A** (6 tests across 3 classes): `TestStatcastGBMTrain`, `TestStatcastGBMPredict`, `TestDefaultModeIsTrueTalent` all train an identical live model on 2 seasons. Could share 1 fixture, eliminating 5 redundant `train()` calls.
- **Group B** (3 tests across 2 classes): `TestStatcastGBMPreseasonTrain`, `TestStatcastGBMPreseasonPredict` all train an identical preseason model on 2 seasons. Could share 1 fixture, eliminating 2 redundant calls.
- **Spy tests** (25 test methods across 14 classes): Each uses `monkeypatch` to spy on an internal function but still executes the full expensive operation end-to-end. These are the largest time sink. Where possible, replace the spy-and-run-through pattern with either (a) a class-scoped fixture that captures the args during a single run, or (b) a recording fake (like `composite/test_model.py`'s `RecordingGBMEngine`) that avoids the expensive operation entirely.

### Steps

1. Create a module-level or class-scoped `live_train_result` fixture that trains a `StatcastGBMModel` on 2 seasons of batter+pitcher data once. Refactor `TestStatcastGBMTrain`, `TestStatcastGBMPredict`, and `TestDefaultModeIsTrueTalent` to use it.
2. Create a similar `preseason_train_result` fixture. Refactor `TestStatcastGBMPreseasonTrain` and `TestStatcastGBMPreseasonPredict` to use it.
3. For spy test classes that verify argument forwarding (sample weights, transforms, per-type params, n_repeats, correlation threshold, min-activity filters, config-top, war-rank-column): evaluate each class and either:
   - (a) Run the expensive operation once in a class-scoped fixture that captures args via a spy, then share the captured args across assertion tests in the class, or
   - (b) Replace the spy pattern with a recording fake that short-circuits the ML work (preferred when feasible).
4. Remove the inline `ablate()` call from `test_multi_holdout_with_validate` by extending the existing `multi_holdout_result` fixture or adding a second class-scoped fixture with validation enabled.
5. Run the full suite and verify. Run `--durations=30` and compare against the baseline.

### Acceptance criteria

- No test in the file calls `model.train()` inline — all training goes through shared fixtures (except tests that need genuinely unique model configurations).
- Spy tests that only verify argument passing do not execute full ML training end-to-end (either via shared fixture or recording fake).
- All 120 tests still pass.
- The top-30 durations list shows measurable improvement for this file.

## Phase 4: Add `pytest-timeout` and CLAUDE.md guidelines

Add per-test timeout enforcement and document test performance expectations to prevent regressions.

### Context

There is no mechanism to prevent a new slow test from landing. CI runs the full suite but doesn't flag individual slow tests. The `CLAUDE.md` code style section has no test performance guidance, meaning AI-assisted development has no guardrails against introducing redundant ML training in tests.

### Steps

1. Add `pytest-timeout` to dev dependencies in `pyproject.toml`.
2. Set a default timeout in `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   timeout = 5
   ```
   This fails any single test that exceeds 5 seconds. Tests marked `@pytest.mark.slow` that legitimately need more time can use `@pytest.mark.timeout(30)`.
3. Add `--durations=10 --durations-min=1.0` to the CI pytest command so slow tests are always visible in PR logs.
4. Add a "Test Performance" section to `CLAUDE.md` with these guidelines:
   - Tests that train ML models must use `@pytest.mark.slow` and share expensive results via `@pytest.fixture(scope="class")`.
   - Never train a model inline in a test method if the result could be shared with sibling tests.
   - No individual test should exceed 5 seconds.
   - Use fakes/recording engines for tests that only verify argument passing or delegation.
   - Follow `tests/models/composite/test_model.py` as the reference pattern for expensive ML test fixtures.
5. Run the full suite with the timeout to verify no existing tests exceed the limit (fix any that do).

### Acceptance criteria

- `pytest-timeout` is in dev dependencies and a default `timeout = 5` is configured.
- CI pytest command includes `--durations=10 --durations-min=1.0`.
- `CLAUDE.md` contains a "Test Performance" section with the guidelines above.
- All existing tests pass within the timeout (with `@pytest.mark.timeout` overrides where needed).

## Ordering

Phases are independent and can be implemented in any order, but the suggested sequence is:

1. **Phase 1** first — it's the lowest-effort change (just moving markers) and immediately improves the fast feedback loop.
2. **Phase 2** next — `test_gbm_training.py` has the most straightforward fixture conversions (helper functions already exist, just need to become fixtures).
3. **Phase 3** after phase 2 — `statcast_gbm/test_model.py` is the largest file and the spy-test refactoring requires more design thought. Lessons from phase 2 will inform the approach.
4. **Phase 4** last — the timeout and guidelines lock in the improvements from phases 1-3 and prevent regressions. It's best to calibrate the timeout after the slow tests are fixed.
