# Feature Store Migration

## Overview

Eliminate duplicate per-year `{player_id: stats}` dict constructions by routing all data lookups through the shared `FeatureStore`. The `FeatureStore` class and its wiring into `PipelineBuilder` are complete; 7 pipeline stages already accept and use an optional `feature_store` parameter. What remains is extending the store to training-time code and evaluating whether `SkillDeltaComputer` should adopt it.

---

## What's Done

| Component | Status | Notes |
|-----------|--------|-------|
| `pipeline/feature_store.py` | Complete | Lazy per-year cached lookups for 4 data types |
| `StatcastRateAdjuster` | Complete | Delegates `batter_statcast(year-1)` to store |
| `BatterBabipAdjuster` | Complete | Delegates `batter_statcast(year-1)` to store |
| `PitcherStatcastAdjuster` | Complete | Delegates `pitcher_statcast(year-1)` to store |
| `PitcherBabipSkillAdjuster` | Complete | Delegates `pitcher_batted_ball(year-1)` to store |
| `GBResidualAdjuster` | Complete | Delegates all 4 lookups to store |
| `MTLBlender` | Complete | Delegates all 4 lookups to store |
| `MTLRateComputer` | Complete | Delegates batter/pitcher statcast + skill lookups |
| `PipelineBuilder` wiring | Complete | Creates one `FeatureStore`, shares across all consuming stages |
| Test coverage | Complete | 11 unit tests + 7 stage integration tests + 2 builder tests |
| `ResidualModelTrainer` | Complete | Delegates batter/pitcher statcast, batted ball, skill lookups to store |

---

## Remaining Steps

### Step 1: Add `pitcher_skill` Lookup to FeatureStore — Skipped

**Status:** Skipped — no consumers in the codebase. Neither `GBResidualAdjuster` nor `MTLBlender` loads pitcher skill data via the feature store path.

---

### Step 2: Adopt FeatureStore in `ml/training.py` — Complete

**File:** `src/fantasy_baseball_manager/ml/training.py` (lines ~80-130)

The GB residual training pipeline builds ~6 dict comprehensions for statcast, batted ball, and skill data across multiple years. These are identical to what `FeatureStore` caches. Accept an optional `feature_store` parameter in `GBResidualTrainer`:

```python
# Before
statcast_data = self.statcast_source.batter_expected_stats(year)
statcast_lookup = {s.player_id: s for s in statcast_data}

# After
if self.feature_store is not None:
    statcast_lookup = self.feature_store.batter_statcast(year)
else:
    statcast_data = self.statcast_source.batter_expected_stats(year)
    statcast_lookup = {s.player_id: s for s in statcast_data}
```

**Acceptance criteria:**
- `GBResidualTrainer` accepts optional `feature_store` parameter
- When provided, all 6 lookups delegate to the store
- Fallback to direct sources when `feature_store` is `None`
- Tests in `tests/ml/` continue to pass

---

### Step 3: Adopt FeatureStore in `ml/mtl/dataset.py`

**File:** `src/fantasy_baseball_manager/ml/mtl/dataset.py` (lines ~60-120)

The MTL dataset builder constructs ~8 dict comprehensions for multi-year feature assembly. Same pattern as Step 2 — accept an optional `feature_store` and delegate lookups:

```python
# Before (repeated for each year in range)
statcast = self.statcast_source.batter_expected_stats(year)
lookup = {s.player_id: s for s in statcast}

# After
if self.feature_store is not None:
    lookup = self.feature_store.batter_statcast(year)
else:
    statcast = self.statcast_source.batter_expected_stats(year)
    lookup = {s.player_id: s for s in statcast}
```

Multi-year caching is already handled — `FeatureStore` caches each year independently, so iterating over `range(2019, 2024)` builds 5 cached entries.

**Acceptance criteria:**
- `MTLDatasetBuilder` accepts optional `feature_store` parameter
- When provided, all lookups delegate to the store
- Tests in `tests/ml/mtl/` continue to pass

---

### Step 4: Wire FeatureStore into Training CLI

**File:** `src/fantasy_baseball_manager/ml/cli.py`

Create a `FeatureStore` in the CLI training commands and pass it to both trainer and dataset builder. This is where the caching payoff is largest — training iterates over multiple years and the same data gets loaded repeatedly.

```python
store = FeatureStore(
    statcast_source=statcast_source,
    batted_ball_source=batted_ball_source,
    skill_data_source=skill_data_source,
)
trainer = GBResidualTrainer(..., feature_store=store)
```

**Acceptance criteria:**
- `ml train` and `ml train-mtl` create a shared `FeatureStore`
- Training commands run successfully with store wired in
- No change to model output (same predictions)

---

### Step 5: Evaluate SkillDeltaComputer Migration

**File:** `src/fantasy_baseball_manager/pipeline/stages/skill_delta_computer.py`

`SkillDeltaComputer` loads skill data for two years (`year-2` and `year-1`) via its own `SkillDataSource` protocol. It's used by `SkillChangeAdjuster`. Evaluate whether it should delegate to `FeatureStore.batter_skill()`:

- **Pro:** Eliminates 2 more dict comprehensions; shares cache with other stages
- **Con:** `SkillDeltaComputer` has a clean, self-contained protocol; adding `FeatureStore` couples it to the pipeline layer

Recommendation: Only migrate if profiling shows the skill data load is a bottleneck. The `SkillDeltaComputer` is already efficient and its protocol boundary is valuable.

**Acceptance criteria:**
- Decision documented (migrate or skip)
- If migrated: `SkillChangeAdjuster` passes feature store through to `SkillDeltaComputer`
- If skipped: no changes needed

---

### Step 6: Remove Fallback Paths (Optional, Long-term)

Once all callers use `FeatureStore`, the `else` branches in each stage become dead code. Consider removing them to simplify the codebase:

- Remove `feature_store is None` fallback in all 7 stages
- Make `feature_store` a required parameter (not optional)
- Update `PipelineBuilder` to always provide a store

**Risk:** This removes the ability to construct stages without a `FeatureStore` in tests. Mitigate by providing a test helper that creates a minimal `FeatureStore` from fake sources.

**Acceptance criteria:**
- All stages require `feature_store` in constructor
- No `if self.feature_store is not None` branches remain
- Test helper `make_test_feature_store(...)` exists for test convenience
- Full test suite passes

---

## Implementation Order and Risk

| Step | Effort | Risk | Dependency |
|------|--------|------|------------|
| 1. Add `pitcher_skill` lookup | Small | Low | Skipped — no consumers |
| 2. Adopt in `ml/training.py` | Medium | Low | Complete |
| 3. Adopt in `ml/mtl/dataset.py` | Medium | Low | None |
| 4. Wire into training CLI | Small | Low | Steps 2-3 |
| 5. Evaluate `SkillDeltaComputer` | Small | None (decision only) | None |
| 6. Remove fallback paths | Large | Medium (breaks test patterns) | Steps 1-4 |

Steps 1-3 can be done in parallel once Step 1 is complete. Step 5 is independent. Step 6 is optional long-term cleanup and should only be done after all other steps are verified and stable.

---

## Verification

After each step:
1. `uv run pytest` — all existing tests pass
2. `uv run pytest tests/pipeline/test_feature_store.py` — store tests pass
3. `uv run ruff check src tests` — no lint issues
4. `uv run ty check src tests` — no type errors

After all steps:
5. Profile a full pipeline run to confirm reduced data loading (each source called once per year instead of once per stage per year)
