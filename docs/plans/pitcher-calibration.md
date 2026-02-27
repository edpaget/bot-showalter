# Pitcher Calibration Roadmap

Fix systematic ERA/FIP bias and range compression in the `statcast-gbm-preseason` pitcher model. The model currently overpredicts ERA by +0.3–0.7 runs across all historical seasons (2017–2025) and compresses predictions into a narrow band (min ERA ~2.8, vs actual elite pitchers at 1.2–2.5). The existing calibration infrastructure (built in top-300-tuning Phase 5) doesn't work in practice — affine calibration amplifies errors (slope 3.25x is unstable), isotonic calibration overfits to the single holdout year, and neither was ever enabled in the production config.

This roadmap also fixes a predict-pipeline bug that silently drops future-season predictions when run alongside historical seasons.

## Observed bias (2025 holdout, `latest` version, 50+ IP)

| Metric | Model | Actual | Bias |
|--------|------:|-------:|-----:|
| Mean ERA | 4.05 | 3.78 | +0.27 |
| Min ERA | 2.72 | 1.17 | +1.55 |
| ERA range | 2.72–5.20 | 1.17–8+ | compressed |

For comparison, Steamer's 2026 min ERA is 2.39; our model's is 2.83.

## Status

| Phase | Status |
|-------|--------|
| 1 — Fix multi-season predict fallback | done (2026-02-27) |
| 2 — Multi-fold calibration | not started |
| 3 — Evaluate bias-correction methods | not started |
| 4 — Pitcher regularization review | not started |

## Phase 1: Fix multi-season predict fallback

The predict pipeline silently drops future seasons when run alongside historical ones.

### Context

`_materialize_with_fallback()` in `model.py` returns immediately if the feature set produces *any* rows. When predicting for `[2017, ..., 2026]`, the feature set returns rows for 2017–2025 (from the stats-table spine) but nothing for 2026 (no stats yet). Because the rows list is non-empty, the `PlayerUniverseProvider` fallback never triggers and 2026 is silently excluded. The user sees "17296 predictions saved" with no indication that 2026 was dropped.

Running `--season 2026` alone works because the feature set returns zero rows, triggering the fallback. But this is error-prone and surprising.

### Steps

1. After materialization, check which of the requested seasons are present in the returned rows.
2. If any seasons are missing and a `PlayerUniverseProvider` is available, collect player IDs for the missing seasons only.
3. Build a supplemental feature set with the missing-season player IDs in the spine filter.
4. Materialize the supplemental set and merge its rows with the original rows.
5. Add a test that predicts for `[current_year, future_year]` and asserts both seasons appear in the output.

### Acceptance criteria

- `predict --season 2025 --season 2026` produces predictions for both seasons in a single run
- Existing single-season fallback behavior is preserved
- No regression in predict performance for seasons that don't need the fallback
- Test coverage for the mixed-season case

## Phase 2: Multi-fold calibration

Replace the single-holdout calibration with cross-validated calibrators fit across multiple years.

### Context

The current calibration fits on a single holdout year (the last season in the training window, typically N≈600 pitchers). This is too few data points for a reliable mapping, especially at the extremes where only a handful of elite pitchers exist. When we tried enabling it:

- **Affine** (slope=3.25, intercept=-8.16): a 1-point raw ERA difference becomes a 3.25-point calibrated difference. Bias worsened from +0.27 to +1.21.
- **Isotonic**: learned a floor at 2.88 ERA and wildly stretched the upper end. Bias worsened to +1.05.

Both overfit to the 2025 holdout distribution and generalized badly to other years.

The `temporal_expanding_cv` infrastructure already exists for hyperparameter tuning. The same fold structure can produce calibrators fit on multiple test sets and aggregated for robustness.

### Steps

1. Add a `fit_multifold_calibrators()` function to `calibration.py` that accepts a list of (raw_predictions, actuals) pairs from multiple CV folds.
2. For affine: average the per-fold slopes and intercepts to produce an aggregated calibrator. Report per-fold and aggregated parameters.
3. For isotonic: pool all fold (predicted, actual) pairs into a single dataset and fit one isotonic regression on the combined data.
4. Wire into `train()`: when `calibrate=true`, use the existing `temporal_expanding_cv` folds to fit per-fold calibrators, then aggregate. Save the aggregated calibrator to the artifact directory.
5. The prediction path (`predict()`) already loads and applies calibrators from the artifact directory — no changes needed there.
6. Add tests verifying that multi-fold calibrators produce more stable parameters than single-fold (e.g., lower variance in slope across leave-one-year-out folds).

### Acceptance criteria

- Multi-fold calibrators use 3+ holdout years of data (not just 1)
- Aggregated affine slope is between 0.8 and 1.5 (not 3.25)
- Calibrated predictions generalize across years (bias consistent, not worse than uncalibrated on any individual year)
- Existing single-fold calibration still works as a fallback for models with <3 training seasons

## Phase 3: Evaluate bias-correction methods

Systematically evaluate which correction method (if any) actually improves predictions on held-out data.

### Context

Three candidate methods, in order of simplicity:

1. **Mean shift**: subtract the average bias from all predictions. The bias is fairly stable at +0.3–0.4 across years. This can't hurt ranking accuracy and is trivially reversible.
2. **Affine (multi-fold)**: linear correction fit from Phase 2. Adjusts both bias and scale.
3. **Isotonic (multi-fold)**: non-parametric correction from Phase 2. Most flexible but highest risk of overfitting.

The evaluation must use truly held-out data — calibrators fit on folds 2017–2024, evaluated on 2025 predictions vs actuals. The current approach of fitting on 2025 and evaluating on 2025 is circular.

### Steps

1. Implement a `MeanShiftCalibrator` in `calibration.py` with the same `.predict()` interface as `AffineCalibrator`.
2. Add `"mean_shift"` as a third `calibration_method` option.
3. Run a comparison experiment:
   - Train on 2017–2024 (holdout 2024 for calibrator fitting via multi-fold CV on 2017–2023).
   - Generate predictions for 2025.
   - Apply each calibration method to the 2025 predictions.
   - Compare RMSE, MAE, and bias against 2025 actuals for: uncalibrated, mean-shift, affine, isotonic.
   - Evaluate on both full-population and top-300 subsets.
4. Select the method with the best bias reduction that doesn't increase RMSE. If no method improves over uncalibrated, document the finding and skip calibration.
5. If a method wins, enable it in `fbm.toml` and update top-300-tuning Phase 5 status to reflect the actual outcome.

### Acceptance criteria

- Comparison table produced for all four methods (uncalibrated + three corrections) on both full-pop and top-300
- Winning method (if any) reduces mean ERA bias to <+0.15 without increasing RMSE by more than 5%
- If no method wins, document the negative result and leave calibration disabled
- Go/no-go decision recorded in the status table

## Phase 4: Pitcher regularization review

Investigate whether the model's hyperparameters are too conservative for predicting elite pitcher performance.

### Context

The current pitcher config (`learning_rate=0.05, max_depth=3, max_leaf_nodes=15, min_samples_leaf=20`) was tuned for best top-300 RMSE across all pitcher stats simultaneously. But this may over-regularize ERA/FIP specifically — the model's predicted ERA range (2.8–5.7) is much narrower than reality (1.2–8+), suggesting the trees can't express enough variation.

The top-300-tuning Phase 3 grid search found these parameters optimal for the *combined* pitcher target RMSE. But ERA and WHIP may have conflicting regularization needs — ERA benefits from more expressiveness (wider range) while WHIP benefits from stability (narrow range, already matches Steamer).

### Steps

1. Run per-target hyperparameter tuning: instead of optimizing for aggregate pitcher RMSE, tune ERA/FIP independently and compare the optimal parameters to the current joint optimum.
2. If ERA-optimal params differ significantly (e.g., deeper trees, higher learning rate), evaluate the trade-off: how much does ERA improve vs how much do other stats degrade?
3. Consider per-target model configs if the trade-off is favorable — train ERA/FIP with one set of hyperparameters and K/9, BB/9, WHIP with another.
4. Evaluate alternative loss functions: `HistGradientBoostingRegressor` supports `loss="absolute_error"` (L1), which is less sensitive to outliers and may reduce compression. Compare L1 vs L2 on ERA holdout metrics.
5. If per-target configs or alternative losses improve ERA without degrading other stats, update `fbm.toml`.

### Acceptance criteria

- Per-target tuning results documented (ERA-optimal vs joint-optimal parameters)
- If per-target configs adopted: ERA prediction range widens (min predicted ERA ≤ 2.5) without WHIP degradation >5%
- If alternative loss adopted: ERA RMSE improves on top-300 without regression on full population
- Go/no-go decision recorded in the status table

## Ordering

**Phase 1** is a standalone bug fix with no model-quality implications — implement first to unblock reliable multi-season predictions.

**Phases 2–3** are sequential: Phase 2 builds the multi-fold calibration infrastructure, Phase 3 evaluates whether it actually helps. Phase 3 may conclude that calibration isn't worth enabling (negative result), which is a valid outcome.

**Phase 4** is independent of Phases 2–3 and addresses the root cause (model expressiveness) rather than the symptom (post-hoc correction). It can run in parallel with or after Phase 3. If Phase 3 finds that calibration helps, Phase 4 may still be worth pursuing to reduce the *need* for calibration.

This roadmap is related to [Top-300 Tuning](top-300-tuning.md) — Phase 5 of that plan (residual analysis and calibration) is marked done but the calibration was never enabled in production. Outcomes from Phases 2–3 here should be reflected back in that plan's status.
