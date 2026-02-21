# Composite GBM Tuning & Improvement

Close the accuracy gap between composite-full GBM and professional projection systems (Steamer, ZiPS). The engine is functional but undertrained and untuned — this roadmap addresses hyperparameter tuning, feature engineering, evaluation methodology, and prediction pipeline gaps.

## Current state (Feb 2025 benchmark)

### All-player performance (2025 season)

| Stat | composite-full (GBM) | Steamer | Gap |
|------|---------------------|---------|-----|
| avg  | 0.064 (R²=0.73) | 0.051 (R²=0.83) | 1.25x |
| obp  | 0.072 (R²=0.79) | 0.055 (R²=0.87) | 1.31x |
| slg  | 0.095 (R²=0.76) | 0.076 (R²=0.85) | 1.25x |
| woba | 0.068 (R²=0.80) | 0.053 (R²=0.88) | 1.28x |

### Top-300 performance (fantasy-relevant players)

| Stat | composite-full (GBM) | Steamer | Gap |
|------|---------------------|---------|-----|
| avg  | 0.093 | 0.071 | 1.31x |
| obp  | 0.100 | 0.071 | 1.41x |
| slg  | 0.128 | 0.086 | 1.49x |
| era  | 2.31  | 1.14  | 2.03x |
| whip | 0.51  | 0.19  | 2.69x |

### Key findings from ablation

- **Statcast-GBM preseason rates are the most valuable non-historical features** — `sc_pre_woba` (+0.040), `sc_pre_era` (+0.059)
- **MLE features contribute zero importance** — all pruned
- **League average rates are noise** — all pruned
- **Pitching model struggles** on top-300 cohort — 2x Steamer on ERA, 2.7x on WHIP
- **Marcel engine doesn't produce woba** — NULL values cause misleading comparisons

## Goal

Reduce RMSE gap to Steamer to ≤1.1x on batting rate stats for top-300 players. Pitching target: ≤1.5x Steamer on ERA/FIP for top-300.

## Status

| Phase | Status |
|-------|--------|
| 1 — Evaluation infrastructure | not started |
| 2 — Hyperparameter tuning | done |
| 3 — Training data expansion | not started |
| 4 — Feature engineering | not started |
| 5 — Model architecture improvements | not started |
| 6 — Top-300 calibration | not started |
| 7 — Continuous evaluation framework | not started |

## Phases

### Phase 1: Evaluation infrastructure

Before tuning, fix the evaluation pipeline so we can trust the numbers.

**1a. Add IP/PA minimum filter to `compare`.**
The `compare` command currently includes all matched players regardless of playing time. Low-IP pitchers with extreme rate stats (ERA 72.00 at 0.1 IP) dominate the pitching RMSE. Add `--min-pa` and `--min-ip` options that filter actuals before computing metrics. Default to 0 (current behavior) for backwards compatibility.

**1b. Add woba derivation to `composite_projection_to_domain`.**
The conversion layer computes avg/obp/slg/ops from counting stats via `_compute_batter_rates` but never derives woba. Add woba computation using FanGraphs linear weights (season-specific wBB, w1B, w2B, w3B, wHR, wHBP). This requires storing a woba-weights table or using a fixed approximation. Both the Marcel and GBM engines feed through `composite_projection_to_domain`, so this fixes woba for all composite variants.

**1c. Normalize PT in rate-stat evaluation.**
When comparing rate-stat accuracy across systems, PT differences can confound the comparison. A system that projects a fringe player at 50 PA and another at 500 PA will appear more accurate on counting stats even if its rate predictions are identical. Add `--normalize-pt consensus` option to `compare` that weights each player's error by their actual PA/IP, giving more weight to players who actually played.

### Phase 2: Hyperparameter tuning ✅

**2a. Wire `tune` operation to composite model.** ✅
Extracted `build_cv_folds` shared helper into `gbm_training.py`, refactored statcast-gbm to use it, added `tune()` method and `DEFAULT_PARAM_GRID` to `CompositeModel`, updated `supported_operations`.

**2b. Per-target hyperparameters.** ✅
Already supported — `GBMEngine.train()` reads `model_params.get("batter", model_params)` / `model_params.get("pitcher", model_params)`.

**2c. Tune and record optimal parameters.** ✅
Full grid search (1024 combos × 2 folds, ~40 min). Optimal params recorded in `fbm.toml`.

**Consensus playing time.** During evaluation, discovered that `proj_pa`/`proj_ip` was NULL for all players because the `playing_time` OLS model only covered seasons 2024-2025. Replaced with consensus PT (Steamer+ZiPS average) via `scripts/generate_consensus_pt.py`, giving 1200+ batters and 1600+ pitchers per season.

#### Phase 2 results: top-300 rate-only (min 200 PA / 50 IP, season 2024)

| Stat | Steamer | ZiPS | v1 (defaults) | v2 (tuned) | v2 vs v1 |
|------|---------|------|---------------|------------|----------|
| avg | 0.0248 | 0.0246 | 0.0570 | **0.0538** | -5.6% |
| obp | 0.0274 | 0.0269 | 0.0669 | **0.0611** | -8.7% |
| slg | 0.0531 | 0.0532 | 0.1039 | **0.1013** | -2.5% |
| ops | 0.0749 | 0.0743 | 0.1596 | **0.1524** | -4.5% |
| woba | 0.0295 | 0.0293 | 0.0672 | **0.0640** | -4.8% |
| era | 1.0441 | 1.1125 | 2.4118 | **1.8224** | -24.4% |
| fip | 0.9151 | 0.9239 | 1.6763 | **1.4755** | -12.0% |
| whip | 0.2166 | 0.2045 | 0.4799 | **0.3979** | -17.1% |
| bb/9 | 0.8566 | 0.8913 | 1.7499 | **1.4715** | -15.9% |
| k/9 | 1.3638 | 1.3240 | 2.8088 | 2.8426 | +1.2% |

Tuning improved all rate stats by 3-24% except K/9. Pitcher stats (ERA -24%, WHIP -17%, BB/9 -16%) saw the largest gains. Gap to Steamer remains ~2x on batting, ~1.5-2x on pitching — closing this further requires feature engineering (Phase 4) and architecture work (Phase 5).

**Known issue:** OPS is still computed via a lossy counting-stat round-trip (`rates → batter_rates_to_counting → _compute_batter_rates`), introducing ~8% error vs the simple `obp + slg`. The `wrc_plus` stat is never computed in the composite pipeline.

### Phase 3: Training data expansion

Currently training on 2 seasons (2022-2023, holdout 2024). With ~1500 batters/season, that's ~3000 training rows — tight for a GBM with 100+ features.

**3a. Expand training window to 4-5 seasons.**
Train on 2019-2023 (excluding 2020 shortened season or weighting it down). This triples the training data. Requires that all feature sources (statcast-gbm preseason, playing time projections) are available for those seasons.

**3b. Backfill upstream projections.**
The statcast-gbm preseason features (`sc_pre_*`) are only populated for seasons where `statcast-gbm-preseason` has been run. Run the upstream model for 2019-2023 to populate these features for the expanded training window. Same for MLE if it ever becomes useful.

**3c. Handle 2020 season.**
The COVID-shortened 2020 season (60 games) has compressed counting stats. Either exclude it, or add a season-length normalization feature so the GBM can learn the adjustment.

### Phase 4: Feature engineering

The ablation showed that many current features are noise. Prune dead weight and add high-signal features.

**4a. Drop zero-importance feature groups.**
Remove `mle_batter_rates` and `league_*_rate` groups from composite variants in `fbm.toml`. These were pruned in ablation with zero importance and add noise. Create a `composite-lean` variant with only the useful groups:
```toml
[models.composite-lean.params]
engine = "gbm"
feature_groups = ["age", "projected_batting_pt", "projected_pitching_pt", "batting_counting_lags", "pitching_counting_lags", "statcast_gbm_preseason_batter_rates", "statcast_gbm_preseason_pitcher_rates"]
```

**4b. Add batted-ball and plate-discipline features.**
Statcast batted-ball data (barrel rate, hard-hit rate, launch angle, sprint speed) and plate discipline metrics (chase rate, whiff rate, zone contact) are strong predictors of batting rate stats. These exist in the statcast database but aren't currently wired as composite feature groups. Add new feature groups:
- `statcast_batted_ball_batter` — barrel%, hard_hit%, avg_launch_angle, avg_exit_velo
- `statcast_plate_discipline` — chase_rate, whiff_rate, zone_contact_rate, swing_rate
- `sprint_speed` — sprint speed percentile

**4c. Add pitcher stuff+ and command features.**
Pitcher-specific Statcast features (stuff+, location+, pitching+, average fastball velocity) would help differentiate pitchers beyond their rate-stat history. Add as a new feature group.

**4d. Interaction features.**
Consider adding engineered interactions: age × performance trend, PT × rate stability. These capture non-linearities that the GBM might struggle with in the raw feature space. Evaluate via ablation before committing.

### Phase 5: Model architecture improvements

**5a. Separate models for starters vs relievers.**
Starting pitchers and relief pitchers have fundamentally different usage patterns and rate-stat distributions. The current single pitcher model conflates them. Split into separate batter/starter/reliever pipelines with role-specific features (GS/G ratio, saves, holds).

**5b. Quantile regression for uncertainty estimates.**
Replace `HistGradientBoostingRegressor` with quantile regression to produce prediction intervals (e.g., 10th/50th/90th percentile). This enables risk-aware draft strategies (prefer low-variance players in head-to-head leagues).

**5c. Multi-season weighting.**
The current training treats all seasons equally. Steamer/ZiPS weight recent seasons more heavily. Add sample weights to `fit_models()` that decay with age of the training season (e.g., 2023 weight=1.0, 2022=0.8, 2021=0.6).

### Phase 6: Top-300 calibration

The biggest accuracy gap is on the top-300 cohort where all players are similar caliber.

**6a. Analyze residual patterns.**
Run `fbm report residuals composite-full/v1 --season 2025 --top 300` to check for systematic bias (e.g., over-projecting power, under-projecting contact). If residuals correlate with specific player attributes, add those as features or apply post-hoc calibration.

**6b. Train on top-N subset.**
Experiment with training only on players with ≥300 PA (batters) or ≥50 IP (pitchers) to focus the model on fantasy-relevant players. The current model wastes capacity learning about fringe players who won't be rostered.

**6c. Isotonic calibration.**
Apply isotonic regression as a post-processing step to calibrate predictions to the observed distribution of rate stats among top-300 players. This corrects for systematic compression (predicting too close to the mean) without changing the underlying model.

### Phase 7: Continuous evaluation framework

**7a. Automated backtesting.**
Create a `backtest` command that trains on years [Y-4..Y-1], predicts year Y, evaluates against actuals, and repeats for multiple years. This gives a more robust accuracy estimate than single-season evaluation and detects overfitting.

**7b. Track metrics over time.**
Store evaluation metrics per model version in the `model_run` table. Add a `fbm runs compare` command that shows accuracy trends across versions, making it easy to see if tuning changes are actually improving predictions.

## Risks and open questions

1. **Steamer/ZiPS have decades of refinement.** Closing the gap fully may require features or modeling approaches we don't yet have. The 1.1x target is ambitious — 1.2x may be a more realistic near-term goal.

2. **Overfitting risk increases with more features.** Batted-ball and plate-discipline features add signal but also dimensionality. Aggressive regularization and cross-validation are essential.

3. **Upstream model quality matters.** The statcast-gbm preseason features are only as good as the statcast-gbm model. Improving that model (separate roadmap) would improve composite-full transitively.

4. **2020 season handling** could introduce noise if not handled carefully. Excluding it is safest but loses data.

5. **Starter/reliever split** requires a reliable role classification feature. Pitchers who change roles mid-season or between seasons are tricky.
