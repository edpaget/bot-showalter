# Generalized Holdout Sampling

Extract the duplicated season-based holdout splitting and evaluation logic from individual models into a shared sampling module, so all models use consistent temporal splits, k-fold CV, and holdout metrics.

## Status

- Phase 1: Done
- Phase 2: Done
- Phase 3: Done
- Phase 4: Done

## Context

Two models already implement holdout sampling with nearly identical strategies:

- **PlayingTime** manually filters rows by season (`model.py:205-222`) and has season-based k-fold CV baked into `select_alpha()` (`engine.py:104-168`).
- **StatcastGBM** splits via `assembler.split()` at the handle level (`model.py:88-89`) and computes holdout metrics in `score_predictions()`.

Both use "last season = holdout, rest = train." The splitting logic is duplicated; the evaluation metrics (R², RMSE) are reimplemented per model. Generalizing the *splitting* and *metric computation* is straightforward. The fit-then-predict step stays model-specific because each model has different signatures.

## Phase 1 — Shared Splitting Utilities

Create `src/fantasy_baseball_manager/models/sampling.py` with pure functions for temporal data splitting.

### Functions

**`temporal_holdout_split`**
- Input: `rows: list[dict[str, Any]]`, `season_column: str = "season"`
- Output: `tuple[list[dict], list[dict]]` — (train rows, holdout rows)
- Behavior: holdout = rows from the max season, train = everything else
- Edge case: raises `ValueError` if fewer than 2 distinct seasons

**`season_kfold`**
- Input: `rows: list[dict[str, Any]]`, `n_folds: int = 5`, `season_column: str = "season"`
- Output: `Iterator[tuple[list[dict], list[dict]]]` — yields (train, test) per fold
- Behavior: groups rows by season, assigns seasons to folds via sorted-index modulo (same deterministic strategy as current `select_alpha`), yields one split per fold
- Edge case: if only 1 season, yields nothing (empty iterator)

### Tests

- Verify train/holdout partition is exhaustive and non-overlapping
- Verify holdout contains only the max season
- Verify k-fold yields correct number of folds
- Verify fold assignment is season-grouped (no season appears in both train and test within a fold)
- Verify single-season edge cases

## Phase 2 — Shared Holdout Metrics

Add metric computation to the same `sampling.py` module, decoupled from any model's fit/predict.

### Functions

**`holdout_metrics`**
- Input: `y_actual: ndarray`, `y_pred: ndarray`
- Output: `dict[str, float]` with keys `r_squared`, `rmse`, `n`
- Behavior: same R²/RMSE math currently in `evaluate_holdout()` and `score_predictions()`

### Tests

- Verify perfect predictions give R² = 1.0 and RMSE = 0.0
- Verify known-value cases
- Verify empty arrays return sensible defaults (R² = 0.0, RMSE = 0.0, n = 0)

## Phase 3 — Migrate PlayingTime Model

Replace the manual splitting and metric logic in the playing-time model with calls to the shared module.

### Changes

- **`engine.py` — `select_alpha()`**: Replace the inline season-grouping and fold-assignment logic (lines 119-143) with `season_kfold()`. The function still owns the alpha-search loop and calls `fit_playing_time` per fold — only the splitting is delegated.
- **`engine.py` — `evaluate_holdout()`**: Replace the inline R²/RMSE computation (lines 231-244) with a call to `holdout_metrics()`. The function still owns calling `fit_playing_time` and building `y_pred`.
- **`model.py` — `train()`**: Replace the manual `last_season` row filtering (lines 205-210) with `temporal_holdout_split()`.
- **`model.py` — `ablate()`**: Same replacement for lines 387-391.

### Constraints

- No behavioral changes — existing tests must pass without modification
- `select_alpha` and `evaluate_holdout` remain in the playing-time engine since they compose the generic splitter with playing-time-specific fitting

## Phase 4 — Migrate StatcastGBM Model

The StatcastGBM model already uses `assembler.split()` for handle-level splitting, which is appropriate for its workflow. The migration here is narrower.

### Changes

- **`training.py` — `score_predictions()`**: If it reimplements R²/RMSE internally, replace with `holdout_metrics()`.
- **`model.py` — `train()` / `ablate()`**: The `config.seasons[:-1]` / `[config.seasons[-1]]` split calculation (lines 88-89, 214-215) can be extracted to a small helper or left as-is since `assembler.split()` handles the actual partitioning. Evaluate whether wrapping it adds clarity or just indirection.

### Constraints

- `assembler.split()` stays as the primary split mechanism for StatcastGBM — don't force row-level splitting where handle-level splitting already works
- Existing tests must pass without modification
