# Fast Feedback Loop

Give an autonomous agent a lightweight train-and-evaluate cycle for rapid iteration on feature ideas. The full pipeline (materialize → train all targets → predict → store → compare) takes minutes and produces far more output than needed for a quick directional check. These tools let the agent test a hypothesis in seconds.

## Status

| Phase | Status |
|-------|--------|
| 1 — Single-target quick-eval | done (2026-03-01) |
| 2 — Marginal value estimator | not started |
| 3 — Feature set A/B comparator | not started |

## Phase 1: Single-target quick-eval

Train a model on one target with a candidate feature set and return holdout RMSE + delta vs baseline.

### Context

The current `train` operation fits all 6 batter targets or all 7 pitcher targets even if the agent only wants to know "does adding barrel rate improve SLG prediction?" This wastes time and makes it harder to attribute signal. A single-target mode trains one `HistGradientBoostingRegressor`, evaluates on one holdout fold, and returns the result immediately.

### Steps

1. Create `src/fantasy_baseball_manager/services/quick_eval.py` with a `quick_eval(feature_columns, target, rows_by_season, train_seasons, holdout_season, params)` function. It extracts the specified columns from the provided rows, trains one GBM on the train seasons, scores on the holdout season, and returns RMSE, R², and sample count.
2. Add a `baseline_rmse` parameter — if provided, the result also includes the absolute and percentage delta vs baseline. This lets the agent compare against the current model's performance on the same target without re-running it.
3. Keep the function stateless — it does not save models, update databases, or touch artifacts. It's a pure computation: data in, metrics out.
4. Add a `fbm quick-eval <model> --target slg --season 2022 --season 2023 --season 2024 [--columns col1 col2 col3] [--baseline 0.085]` CLI command. If `--columns` is omitted, use the model's default feature set.
5. Support `--inject <name>` to add a named candidate from the feature-candidate-factory into the feature matrix alongside the model's default columns.
6. Write tests with synthetic data verifying correct RMSE calculation, single-target isolation, and delta reporting.

### Acceptance criteria

- Only one target is trained and evaluated (not all 6/7).
- RMSE matches what the full training pipeline would produce for that target on the same data split.
- Delta vs baseline is correctly computed when a baseline is provided.
- The `--inject` flag integrates a named candidate column into the feature matrix.
- No side effects (no files written, no database changes).

## Phase 2: Marginal value estimator

Estimate the RMSE improvement from adding one candidate feature to the existing feature set.

### Context

The ablation workflow measures the cost of *removing* features. The agent also needs the inverse: the *benefit* of *adding* a feature. This is a fast way to screen candidates — if adding barrel rate to a feature set that already contains hard-hit rate doesn't improve RMSE, the features are redundant and the candidate can be skipped.

### Steps

1. Add a `marginal_value(candidate_column, feature_columns, targets, rows_by_season, train_seasons, holdout_season, params)` function to quick_eval. It trains two models: one with the current feature columns, one with the candidate column appended. Returns per-target RMSE deltas.
2. Use the same train/holdout split for both models to ensure a fair comparison (same random seed, same data).
3. Report: per-target RMSE without candidate, per-target RMSE with candidate, absolute delta, percentage delta, and a summary verdict ("improves N/M targets").
4. Add a `fbm marginal-value <model> --candidate "<expression_or_name>" --seasons 2022 2023 2024 --player-type batter` CLI command.
5. Support testing multiple candidates in one run: `--candidate barrel_ev --candidate chase_rate_sq` evaluates each independently and ranks them by average improvement.
6. Write tests with synthetic data where one candidate is genuinely predictive and another is pure noise, verifying correct ranking.

### Acceptance criteria

- Two models are trained on identical data with the only difference being the candidate column.
- Per-target deltas are correctly attributed to the candidate.
- Multi-candidate mode ranks candidates by average improvement.
- Noise candidates correctly show near-zero or negative marginal value.

## Phase 3: Feature set A/B comparator

Train two models with different feature sets on identical folds and report per-target RMSE deltas.

### Context

The existing `compare` command works at the projection level — it requires predictions to be stored in the database, which involves the full predict → store pipeline. The agent needs a training-level comparison that skips that overhead. This is the tool for answering "is feature set B better than feature set A?" without generating full projections.

### Steps

1. Add a `compare_feature_sets(columns_a, columns_b, targets, rows_by_season, train_seasons, holdout_season, params)` function to quick_eval. Trains two models, evaluates both on the same holdout, and returns a per-target comparison table.
2. Support cross-validated comparison: if 3+ seasons are provided, use temporal expanding CV and average the deltas across folds. This gives a more robust signal than single-holdout comparison.
3. Report: per-target RMSE for A, per-target RMSE for B, delta, percentage delta, and a summary ("B wins N/M targets").
4. Add a `fbm compare-features <model> --set-a "col1,col2,col3" --set-b "col1,col2,col3,col4" --seasons 2022 2023 2024 --player-type batter` CLI command.
5. Support referencing the model's current default feature set as set A via `--set-a default`, so the agent only needs to specify set B.
6. Write tests with synthetic data verifying correct per-target comparison and CV averaging.

### Acceptance criteria

- Both models train on identical data splits.
- Per-target deltas correctly reflect the feature set difference.
- Cross-validated mode averages deltas across temporal folds.
- `--set-a default` correctly loads the model's current curated feature list.
- Results are consistent with what `marginal_value` would report for single-feature additions.

## Ordering

Phases are sequential: 1 → 2 → 3. Phase 1 provides the core single-target evaluation primitive. Phase 2 uses phase 1 internally to compare with/without a candidate. Phase 3 generalizes phase 2 to arbitrary feature set pairs.

## Dependencies

- **feature-candidate-factory (phase 1)**: The `--inject` flag in phase 1 and the `--candidate` flag in phase 2 reference named candidates from the feature factory. Without it, users must provide pre-materialized columns.
