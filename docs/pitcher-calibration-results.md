# Pitcher Calibration: Experiment Results

Summary of the pitcher-calibration roadmap (4 phases, Feb–Mar 2026). The goal was to fix systematic ERA/FIP bias and range compression in the `statcast-gbm-preseason` pitcher model.

**Outcome: no actionable improvement found.** The production config is unchanged. One bug fix shipped (Phase 1), useful infrastructure was built (Phases 2–4), and the investigation narrowed the root cause to feature-set limitations rather than model configuration.

## The problem

The pitcher model compresses ERA predictions into a narrow band:

| Metric | Model | Actual | Gap |
|--------|------:|-------:|----:|
| Mean ERA | 4.05 | 3.78 | +0.27 |
| Min ERA | 2.72 | 1.17 | +1.55 |
| ERA range | 2.72–5.20 | 1.17–8+ | compressed |

Elite pitchers (sub-2.5 ERA) are systematically overpredicted. For context, Steamer's min projected ERA is 2.39; ours is 2.83.

## What we tried

### Phase 1: Fix multi-season predict fallback (GO)

A bug caused `predict --season 2025 --season 2026` to silently drop 2026. Fixed by checking which requested seasons are present after materialization and fetching missing ones via the `PlayerUniverseProvider` fallback.

### Phase 2: Multi-fold calibration (infrastructure)

Replaced single-holdout calibrator fitting with cross-validated calibrators aggregated across 6+ holdout years. The single-holdout approach produced unstable parameters (affine slope of 3.25x) that amplified errors. Multi-fold calibrators are more stable (slope ~1.0–1.5) but still face the fundamental bias-direction problem discovered in Phase 3.

### Phase 3: Post-hoc bias correction (NO-GO)

Tested three correction methods: mean shift, affine, and isotonic regression. All three applied to raw model predictions after training.

**Full population:** All methods improved RMSE and reduced bias (2–5% RMSE reduction).

**Top-300 pitchers:** All methods catastrophically worsened RMSE (+46% to +88%).

**Root cause:** The model's bias flips direction between populations. Full-population bias is negative (underpredicts ERA for the long tail of bad pitchers), while top-300 bias is positive (overpredicts ERA for good pitchers). Any uniform correction that fixes one population breaks the other.

| Method | Full-pop ERA RMSE | Top-300 ERA RMSE |
|--------|------------------:|------------------:|
| Uncalibrated | 6.62 | 1.20 |
| Mean shift | 6.49 (-2.0%) | 2.24 (+87.6%) |
| Affine | 6.44 (-2.7%) | 2.06 (+72.6%) |
| Isotonic | 6.41 (-3.2%) | 1.94 (+62.3%) |

### Phase 4: Hyperparameter regularization (NO-GO)

Re-tuned pitcher hyperparameters with a 486-combination grid search (3 depths x 3 iterations x 3 learning rates x 3 leaf sizes x 3 leaf node limits x 2 loss functions), evaluated on top-300 with expanding CV over 2017–2024.

**Tuning found a different optimum:** `max_depth=7, max_iter=200, loss=absolute_error` (vs production: `max_depth=3, max_iter=100, loss=squared_error`). The new config uses deeper trees and L1 loss. Per-target divergence is small — all pitcher targets prefer similar parameters, with k_per_9 being the most different (+2.5% delta from joint).

**Holdout comparison (2024 + 2025):** The deeper/L1 config consistently reduces top-300 RMSE by 3–8% but degrades ranking accuracy (Spearman rho) by 1–12%.

| Variant | Pop. | Season | RMSE wins | rho wins |
|---------|------|--------|-----------|----------|
| depth=7, L1 | Full | 2025 | 1/5 | 2/5 |
| depth=7, L1 | Top 300 | 2025 | 4/5 | 2/5 |
| depth=7, L1 | Full | 2024 | 1/5 | 2/5 |
| depth=7, L1 | Top 300 | 2024 | 4/5 | 1/5 |
| depth=3, L1 | Full | 2025 | 1/5 | 0/5 |
| depth=3, L1 | Top 300 | 2025 | 4/5 | 1/5 |

**Interpretation:** Deeper trees widen the prediction range (reducing RMSE) but introduce more ranking errors. The production config's conservative regularization trades absolute accuracy for ranking quality — the right trade-off for fantasy baseball, where getting the order right matters more than nailing the exact ERA.

## Conclusions

1. **ERA range compression is a feature-set limitation, not a model configuration problem.** The preseason features (lagged seasonal averages) don't contain enough signal to distinguish a 2.0-ERA pitcher from a 3.0-ERA pitcher before the season starts. More expressive models just overfit to noise.

2. **Bias correction can't work uniformly** because the bias direction depends on pitcher quality tier. Any method that reduces overprediction for top pitchers increases underprediction for the long tail, and vice versa.

3. **Ranking accuracy and absolute accuracy trade off** in this regime. The production config sits at the right point on this trade-off for fantasy use cases.

4. **Future directions** that might help:
   - Per-stratum calibration (separate corrections for top-100 / 100-300 / tail)
   - Additional features that capture elite-pitcher distinctiveness (e.g., pitch-level Statcast metrics, stuff+ models)
   - Larger training windows (more historical data for the tails)

## Infrastructure delivered

Despite the no-go outcomes, the roadmap produced useful infrastructure:

- **Multi-fold calibration** (`fit_multifold_calibrators`) — available for future calibration work
- **Mean shift calibrator** — simplest possible bias correction, ready if per-stratum correction is attempted
- **Per-target tuning analysis** (`extract_per_target_best`) — shows when individual targets want different hyperparameters than the joint optimum
- **L1 loss support** — `fit_models` now accepts `loss` parameter
- **Per-target param overrides** — `per_target_params` in `fit_models` and the train path enable target-specific hyperparameter configs
- **JSON CLI parsing** — `--param` flags accept JSON dicts/lists for complex configurations
- **Multi-season predict fix** — `predict --season 2025 --season 2026` now works correctly
