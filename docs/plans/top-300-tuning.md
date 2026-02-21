# Top-300 Tuning Roadmap

Improve `statcast-gbm-preseason` accuracy on the top-300 fantasy-relevant player cohort. The model currently matches Steamer on the full population but falls behind on top-300 (where all players are similar caliber and differentiation is harder). The gap is largest on batting rate stats (AVG 0.080 vs 0.071, OBP 0.081 vs 0.071) and pitching (ERA 1.75 vs 1.14, WHIP 0.29 vs 0.19).

The root cause is that the model trains on all players equally — fringe players with 50 PA get the same weight as regulars with 600 PA. This wastes model capacity on a population segment irrelevant to fantasy and biases predictions toward the population mean, which hurts top-300 accuracy where the signal lives in the tails.

## Current state (Feb 2026 benchmark)

### Full-population performance (2025 season)

| Stat | statcast-gbm-preseason | Steamer | Ratio |
|------|----------------------|---------|-------|
| avg  | 0.0516 (R²=0.826) | 0.0514 (R²=0.826) | 1.00x |
| obp  | 0.0550 (R²=0.875) | 0.0551 (R²=0.875) | 1.00x |
| slg  | 0.0758 (R²=0.850) | 0.0763 (R²=0.848) | 0.99x |
| woba | 0.0514 (R²=0.885) | 0.0526 (R²=0.880) | 0.98x |
| era  | 6.33 (R²=0.072) | 7.34 (R²=neg) | 0.86x |
| fip  | 3.36 (R²=0.139) | 4.48 (R²=neg) | 0.75x |

### Top-300 performance (2025 season)

| Stat | statcast-gbm-preseason | Steamer | Ratio |
|------|----------------------|---------|-------|
| avg  | 0.0795 (R²=neg) | 0.0710 (R²=neg) | 1.12x |
| obp  | 0.0814 (R²=neg) | 0.0706 (R²=neg) | 1.15x |
| slg  | 0.1033 (R²=neg) | 0.0855 (R²=neg) | 1.21x |
| woba | 0.0744 (R²=neg) | 0.0626 (R²=neg) | 1.19x |
| era  | 1.75 | 1.14 | 1.54x |
| fip  | 1.23 | 0.76 | 1.62x |

### Goal

Reduce RMSE ratio to Steamer to ≤1.05x on batting rate stats (AVG, OBP, SLG, wOBA) for top-300 players. Pitching target: ≤1.3x on ERA/FIP.

## Status

| Phase | Status |
|-------|--------|
| 1 — PA/IP-weighted training | done |
| 1.5 — Configurable weight transforms | done |
| 2 — Min-PA/IP training filter | done |
| 3 — Hyperparameter re-tuning | not started |
| 4 — Expanded training window | not started |
| 5 — Residual analysis and calibration | not started |
| 6 — Feature engineering for top-300 differentiation | not started |

## Phase 1: PA/IP-weighted training

The highest-leverage change. Currently `fit_models()` treats every training row equally. A player with 50 PA and one with 650 PA contribute the same to the loss function. This means the model optimizes for the average player, not the typical fantasy-relevant player.

### Context

`HistGradientBoostingRegressor.fit()` accepts a `sample_weight` parameter. By weighting rows proportional to PA (batters) or IP (pitchers), the model's loss function naturally prioritizes getting high-activity players right. This is how Steamer and ZiPS handle it — they're regressing toward league means using PA as a reliability weight.

The current pipeline has PA and IP available in the training data as lag features (`pa_1`, `ip_1`), but they're only used as input features, not as sample weights.

### Steps

1. Add a `sample_weight` parameter to `fit_models()` in `gbm_training.py`. When provided, pass it through to `model.fit()`.
2. Add a `_batter_sample_weight_column` and `_pitcher_sample_weight_column` property to the model base class, defaulting to `None` (no weighting, preserving current behavior).
3. In `StatcastGBMPreseasonModel`, override these to return `"pa_1"` and `"ip_1"` respectively.
4. Update `train()` to extract sample weights from training rows and pass them to `fit_models()`.
5. Update `grid_search_cv()` to accept and forward sample weights so that hyperparameter tuning also optimizes for the weighted objective.
6. Re-tune batter hyperparameters with the new weighted objective (`fbm tune`).

### Acceptance criteria

- `fit_models()` accepts optional `sample_weight` and passes it to sklearn
- Preseason model training uses PA/IP weighting by default
- Full test suite passes, including existing ablation and model tests
- Top-300 batter RMSE improves relative to baseline (measure via `fbm compare --top 300`)

### Phase 1 results

Raw PA/IP weighting improved top-300 modestly but regressed full-population significantly. The raw weights are too aggressive — a 650-PA player gets 13x the weight of a 50-PA player.

| Stat | Baseline | PA/IP weighted | Delta |
|------|----------|---------------|-------|
| AVG (top-300) | 0.080 | 0.077 | -3.8% |
| ERA (top-300) | 1.75 | 1.70 | -2.9% |
| SLG (full-pop) | 0.076 | 0.092 | +21% |
| wOBA (full-pop) | 0.051 | 0.062 | +22% |

## Phase 1.5: Configurable weight transforms

Compress the sample weight range with transform functions to reduce the aggressiveness of PA/IP weighting. A registry of named transforms (raw, sqrt, log1p, clamp variants) allows automated sweeping to find the best balance between top-300 improvement and full-population stability.

[Phase plan](top-300-tuning/phase-1.5.md)

### Steps

1. Add a `sample_weight_transforms` module with a registry of named transform functions.
2. Add a `sample_weight_transform` parameter to `build_cv_folds()` to apply transforms during fold construction.
3. Add a `sweep_cv()` function to `gbm_training.py` that sweeps meta-parameters (like weight transforms) requiring fold rebuilding.
4. Add `_sample_weight_transform` property and `_resolve_weight_transform()` helper to the model base class.
5. Wire the transform through `train()`, `tune()`, and a new `sweep()` method.
6. Add `fbm sweep` CLI command mirroring `tune`.

### Acceptance criteria

- Transform registry with raw, sqrt, log1p, and clamp variants
- `sweep_cv()` evaluates all transforms and returns best via `GridSearchResult`
- Model `sweep()` returns `TuneResult` with best meta-params per player type
- `fbm sweep` CLI command prints comparison table
- Full test suite passes

## Phase 2: Min-PA/IP training filter

Remove low-activity noise from training data entirely. Players with <50 PA or <10 IP have extreme rate stats that are essentially random — including them adds noise even with PA-weighting.

### Context

The current pipeline includes all players in training regardless of playing time. A reliever who threw 2 IP with a 27.00 ERA trains alongside full-season starters. Even with PA/IP weighting (Phase 1), these rows contribute gradient signal that may hurt generalization for the top-300 cohort.

### Steps

1. Add `_batter_min_pa` and `_pitcher_min_ip` properties to the model base class, defaulting to `0` (include all).
2. In `StatcastGBMPreseasonModel`, set sensible thresholds (e.g., `min_pa=100`, `min_ip=20`) based on the data distribution.
3. Apply the filter in `train()` after reading rows but before extracting features/targets. Filter on the target season's actual PA/IP, not the lagged values.
4. Also apply the filter in `tune()` so hyperparameter search matches training conditions.
5. Evaluate impact on top-300 vs full-population accuracy. If full-population accuracy degrades significantly, make the filter configurable via `model_params`.

### Acceptance criteria

- Training filters out low-activity players
- Top-300 accuracy improves without catastrophic full-population degradation
- Filter thresholds are configurable for experimentation

## Phase 3: Hyperparameter re-tuning

The batter model currently uses sklearn defaults (no tuned hyperparameters), while the pitcher model has tuned params from a prior run. Both need re-tuning under the new weighted, filtered training regime.

### Context

The pitcher model already has tuned params in `fbm.toml`: `learning_rate=0.01, max_depth=3, max_iter=200, max_leaf_nodes=15, min_samples_leaf=50`. The batter model uses defaults. After Phases 1-2 change the training data distribution (via weighting and filtering), the optimal hyperparameters will shift — particularly `min_samples_leaf` and `max_depth`, which control how much the model can specialize for subpopulations.

### Steps

1. Run `fbm tune statcast-gbm-preseason --season 2019 --season 2020 --season 2021 --season 2022 --season 2023 --season 2024 --season 2025` with the weighted/filtered training pipeline.
2. Record optimal parameters for both batters and pitchers.
3. Update `fbm.toml` with new batter and pitcher parameters.
4. Re-evaluate on top-300 and full-population to confirm improvement.

### Acceptance criteria

- Both batter and pitcher models have tuned hyperparameters in `fbm.toml`
- Tuning uses the weighted/filtered objective from Phases 1-2
- Top-300 RMSE improves relative to Phase 2 baseline

## Phase 4: Expanded training window

More training data improves generalization, especially for the top-300 where data is sparse. The model currently trains on 3 seasons (holdout excluded). Expanding to 5+ seasons nearly doubles the training rows.

### Context

The preseason feature builders support arbitrary season lists. The limiting factor is upstream data availability — Statcast data goes back to 2015, but quality improves over time. The 2020 COVID-shortened season (60 games) needs special handling since rate stats have higher variance at lower sample sizes.

### Steps

1. Verify Statcast data availability for 2018-2025 by running the feature builders on those seasons.
2. Handle the 2020 season: either exclude it entirely or downweight it (via sample weight × season_games/162).
3. Train with expanded window: `--season 2018 --season 2019 ... --season 2025`.
4. Re-tune hyperparameters on the expanded dataset (Phase 3 process).
5. Evaluate impact on top-300 accuracy across multiple holdout years (2023, 2024, 2025) to check for overfitting.

### Acceptance criteria

- Model trains on 5+ seasons without errors
- 2020 season handled appropriately (excluded or downweighted)
- Multi-year holdout evaluation shows consistent improvement, not just single-year gains

## Phase 5: Residual analysis and calibration

Analyze systematic prediction errors among top-300 players and apply post-hoc corrections if warranted.

### Context

GBM models tend to compress predictions toward the training mean — they're better at ranking than at getting absolute values right. For the top-300 cohort (which is above-mean by definition), this manifests as systematic underestimation of performance. Understanding the residual structure reveals whether the issue is feature-driven (fixable with better inputs) or model-driven (fixable with calibration).

### Steps

1. Generate predictions for a multi-year test set (e.g., 2023-2025 holdout years).
2. Compute residuals (actual - predicted) for top-300 players per year.
3. Analyze residual patterns: correlate residuals with player attributes (age, PA, position, prior-year performance). Check for systematic bias (mean residual ≠ 0) and heteroscedasticity (variance varies with predicted value).
4. If systematic bias exists, implement isotonic calibration: fit `sklearn.isotonic.IsotonicRegression` on predicted-vs-actual pairs from CV folds, apply as a post-processing step in `predict()`.
5. If residuals correlate with player attributes, consider adding those as features (Phase 6).

### Acceptance criteria

- Residual analysis report generated for top-300 across multiple years
- If calibration applied: top-300 RMSE improves and calibration curve is monotonic
- No degradation on full-population metrics

## Phase 6: Feature engineering for top-300 differentiation

Add features that help differentiate elite players from each other, not just from the population mean.

### Context

Among top-300 players, the current features may lack the granularity needed for differentiation. For example, all top-300 batters have similar exit velocities, so `avg_exit_velo` doesn't help rank them. Features that capture year-over-year trends, consistency, or platoon splits might add signal within this cohort.

### Steps

1. Add trend features: `avg_1 - avg_2` (year-over-year change in batting average), same for OBP, SLG. These capture whether a player is improving or declining, which is more informative than the raw level for elite players.
2. Add consistency features: standard deviation of monthly rate stats within a season (from game logs if available). Consistent performers are more predictable.
3. Add age-interaction features: `age × avg_1` to capture age-related decline curves that differ by performance level.
4. Run ablation on the expanded feature set to identify which new features contribute positively.
5. Prune any new features that don't improve top-300 accuracy.

### Acceptance criteria

- At least one new feature type improves top-300 accuracy via ablation
- Feature additions don't degrade full-population accuracy
- New features have test coverage following established curated-column patterns

## Ordering

Phases should be implemented in order:

1. **Phase 1** (PA/IP weighting) — highest expected impact, simple to implement. No dependencies.
2. **Phase 2** (min-PA/IP filter) — builds on Phase 1's infrastructure. Small incremental change.
3. **Phase 3** (re-tuning) — must follow Phases 1-2 since hyperparameters depend on the training regime.
4. **Phase 4** (expanded training window) — independent of Phases 1-3 but re-tuning (Phase 3) should be repeated after.
5. **Phase 5** (residual analysis) — diagnostic phase, can run anytime after Phase 3 to assess remaining gaps.
6. **Phase 6** (feature engineering) — informed by Phase 5's residual analysis. Highest risk, highest potential payoff.

After each phase, run `fbm compare statcast-gbm-preseason/latest steamer/2025 --season 2025 --top 300` to measure progress against the gap.
