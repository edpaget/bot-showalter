# Feature Store Migration

## Status: Complete

All pipeline stages, training code, and CLI commands now route data lookups through the shared `FeatureStore`. Fallback paths have been removed — `feature_store` is a required parameter everywhere. A shared test helper `make_test_feature_store()` in `tests/conftest.py` provides convenient construction with null stubs for unused source slots.

---

## What's Done

| Component | Status | Notes |
|-----------|--------|-------|
| `pipeline/feature_store.py` | Complete | Lazy per-year cached lookups for 5 data types |
| `StatcastRateAdjuster` | Complete | Requires `feature_store`, delegates `batter_statcast(year-1)` |
| `BatterBabipAdjuster` | Complete | Requires `feature_store`, delegates `batter_statcast(year-1)` |
| `PitcherStatcastAdjuster` | Complete | Requires `feature_store`, delegates `pitcher_statcast(year-1)` |
| `PitcherBabipSkillAdjuster` | Complete | Requires `feature_store`, delegates `pitcher_batted_ball(year-1)` |
| `GBResidualAdjuster` | Complete | Requires `feature_store`, delegates all 4 lookups |
| `MTLBlender` | Complete | Requires `feature_store`, delegates all 4 lookups |
| `MTLRateComputer` | Complete | Requires `feature_store`, delegates batter/pitcher statcast + skill lookups |
| `SkillDeltaComputer` | Complete | Requires `feature_store`, delegates batter_skill + pitcher_skill lookups |
| `ResidualModelTrainer` | Complete | Requires `feature_store`, delegates all lookups |
| `BatterTrainingDataCollector` | Complete | Requires `feature_store`, delegates batter statcast + skill lookups |
| `PitcherTrainingDataCollector` | Complete | Requires `feature_store`, delegates pitcher statcast + batted ball lookups |
| `MTLTrainer` | Complete | Requires `feature_store`, passes through to collectors |
| `PipelineBuilder` wiring | Complete | Creates one `FeatureStore` via `_resolve_feature_store()`, shared across all stages |
| ML CLI wiring | Complete | Constructs `FeatureStore` from sources, passes to trainers |
| Test helper | Complete | `make_test_feature_store()` in `tests/conftest.py` with null stub sources |
| Fallback removal | Complete | No `feature_store is not None` branches remain in src/ |

---

## Verification

All checks pass:
- `uv run pytest` — 2096 tests pass
- `uv run ruff check src tests` — clean
- `uv run ty check src tests` — clean
- `grep -r "feature_store is not None" src/` — zero results (only `_resolve_feature_store()` caching check remains)
