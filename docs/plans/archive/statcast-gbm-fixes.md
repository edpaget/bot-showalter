# Statcast GBM Model Fixes

**Status: Complete**

## Goal

Fix bugs, correctness issues, and code smells identified in a review of the
statcast-gbm model. Organized from most critical (model-breaking) to least
critical (minor improvements).

## Phases

### Phase 1: Fix X/y Length Mismatch in Training Pipeline — `791cf20`

Filter X rows per-target so `extract_features` and `extract_targets` always
align. Updated `fit_models`, `score_predictions`, and
`compute_permutation_importance` to use per-target filtering.

### Phase 2: Fix Statcast Join to Support Pitchers — `a164b27`

Extended `_build_raw_query` to join on `sc.pitcher_id` when
`player_type == "pitcher"`, threaded through `_run_transforms`.

### Phase 3: Validate `config.seasons` Length in `train()` — `3031822`

Added guard raising `ValueError` when `len(config.seasons) < 2` in both
`train()` and `ablate()`.

### Phase 4: Return `NaN` Instead of `0.0` for Missing Transform Data — `3b174e9`

Changed all five statcast transforms and the assembler default fill to return
`float("nan")` for undefined metrics (zero denominator). Computed zeros
(0/n where n > 0) remain `0.0`.

### Phase 5: Remove Unused `plate_x`/`plate_z` Columns and Fix `barrel` Check — `eed1cf9`

Removed unused columns from `PLATE_DISCIPLINE.columns`. Changed barrel
counting from truthiness to explicit `== 1`.

### Phase 6: Replace Hardcoded `_XSLG_SCALE` with Statcast Column — `8a9a04d`

Added `estimated_slg_using_speedangle` to the schema (migration 002), domain,
mapper, and repo. Compute `xslg` as the direct mean of the statcast column
instead of scaling `xwOBA * 1.25`.

## Out of Scope

- **Permutation importance performance** (`deepcopy` in loop): real but
  low-priority; only affects `ablate` wall-clock time, not correctness.
- **Pitch-mix total-count semantics** (non-tracked types excluded): a design
  decision, not a bug. Could revisit if pitch-type coverage expands.
- **Spin profile pitch-type coverage** (3 types vs pitch-mix's 6): adding
  more spin features is a feature request, not a fix.
- **Mutable fields on frozen `ModelConfig`**: cross-cutting concern that
  affects all models, not statcast-specific.
