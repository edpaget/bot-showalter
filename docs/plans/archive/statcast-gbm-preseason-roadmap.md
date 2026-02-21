# Statcast-GBM Preseason Model Roadmap

**Model:** `statcast-gbm-preseason` (`StatcastGBMPreseasonModel`)
**Created:** 2026-02-17
**Status:** Active
**Goal:** Improve the preseason projection model to be competitive with Marcel and
Steamer on batter stats while maintaining its pitcher ERA/FIP/WHIP advantage.

## Model Identity

`statcast-gbm-preseason` is a **projection model** that uses prior-season (lag=1)
Statcast features to make genuine pre-season predictions. It competes directly with
Marcel and Steamer as a blind forecast system.

**Use cases:**
- Standalone preseason projections (especially pitcher rate stats)
- Potential ensemble component for preseason blends
- Alternative to Marcel/Steamer for pitcher ERA/FIP/WHIP in draft prep

**Current strengths:** Leads all systems on pitcher ERA, FIP, and WHIP for the
full player pool. Prior-year stuff metrics (spin, velocity, movement) are highly
stable year-to-year.

**Current weakness:** Trails Marcel and Steamer on batter stats (AVG, OBP, SLG)
and most top-200 metrics. Batter Statcast features lose ~22-46% accuracy when
lagged by one year.

## Current Baseline (unpruned features)

Training: 2022-2024, holdout: 2025. Full unpruned feature sets (35 batter, 46
pitcher). Feature pruning was attempted and reverted after failing the go/no-go
gate (only 2/8 holdout targets improved).

### Holdout RMSE

| Target | RMSE |
|--------|------|
| avg | 0.0492 |
| obp | 0.0589 |
| slg | 0.0873 |
| woba | 0.0583 |
| iso | 0.0512 |
| babip | 0.0744 |
| era | 6.268 |
| fip | 3.467 |
| k/9 | 2.980 |
| bb/9 | 3.757 |
| hr/9 | 5.216 |
| babip (pitcher) | 0.123 |
| whip | 0.923 |

### Evaluation RMSE (full pool, 2025 actuals)

| Stat | Preseason | Marcel | Steamer |
|------|-----------|--------|---------|
| avg | 0.0585 | 0.0587 | **0.0515** |
| obp | 0.0648 | 0.0643 | **0.0551** |
| slg | 0.0863 | 0.0937 | **0.0763** |
| era | **6.626** | 7.670 | 7.342 |
| fip | **3.472** | 12.817 | 4.486 |
| whip | **0.904** | 1.248 | 1.191 |
| k/9 | 2.833 | **2.435** | 2.122 |
| bb/9 | 3.245 | 3.541 | **3.456** |

### Completed Work

- **Model split:** Extracted from shared class into dedicated
  `StatcastGBMPreseasonModel` with its own feature sets and artifact directory.
- **Feature pruning (Phase 1 attempt):** NO-GO — only 2/8 targets improved.
  Reverted to full unpruned features. Pruning may be re-attempted after other
  improvements establish a stronger baseline.

---

## Phase 1: Hyperparameter Tuning

**Rationale:** The model uses scikit-learn defaults. Since feature pruning failed,
hyperparameter tuning is the lowest-risk first improvement. The preseason model's
batter weakness may partly stem from suboptimal regularization — lagged features
are noisier and may benefit from stronger shrinkage.

### Work

1. Add a `tune` operation (or standalone script) that runs time-series
   cross-validation over a parameter grid:
   - `max_iter`: [100, 200, 500, 1000]
   - `max_depth`: [3, 5, 7, None]
   - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
   - `min_samples_leaf`: [5, 10, 20, 50]
   - `max_leaf_nodes`: [15, 31, 63, None]
2. Use the full (unpruned) feature sets.
3. Tune batter and pitcher sub-models separately — the batter model likely needs
   stronger regularization (higher `min_samples_leaf`, lower `max_depth`) given
   the noisier lagged features.
4. Store best params in model config.

### Go/No-Go Gate

- **Metric:** Holdout RMSE and 2025 evaluation RMSE vs current baseline.
- **Go:** Mean RMSE across all targets improves by ≥ 2%, with no single target
  degrading more than 5%.
- **No-go:** Keep defaults and move on. HistGradientBoosting defaults are
  generally strong.

---

## Phase 2: Multi-Year Statcast Averaging

**Rationale:** The biggest weakness is that single-year Statcast metrics are
volatile when lagged. Whiff_rate importance drops ~95% from live mode to preseason
mode. Averaging 2 years of Statcast data should stabilize the signal and recover
some of the information lost to year-over-year noise.

### Work

1. Extend the `TransformFeature` lag system to support multi-year averaging
   (e.g., `BATTED_BALL.with_avg_lag(1, 2)` computes the mean of lag-1 and lag-2
   values).
2. Alternatively, add a new `AveragedTransformFeature` group type that the
   assembler handles by joining multiple lagged seasons and averaging.
3. Build `build_batter_preseason_averaged_set()` and
   `build_pitcher_preseason_averaged_set()` using 2-year averaged Statcast
   features alongside lag-1 traditional stats.
4. Requires expanding training data to at least 3 seasons so 2-year averages are
   available for the training set.

### Go/No-Go Gate

- **Metric:** Preseason holdout RMSE and 2025 evaluation RMSE.
- **Go:** Batter RMSE improves on ≥ 3/6 targets (avg, obp, slg, woba, iso,
  babip). Pitcher targets must not degrade more than 3% on any single stat.
- **No-go:** Try weighted recency-biased averaging (70% lag-1 / 30% lag-2)
  before abandoning.

---

## Phase 3: New Batter Features

**Rationale:** Batter features are the model's structural weakness. In preseason
mode only `pull_pct` among the Statcast transforms shows meaningful importance.
New feature categories could provide more stable year-over-year batter signal.

### Work

Investigate each independently:

1. **Sprint speed** — Baseball Savant sprint speed (ft/s). Highly stable
   year-over-year (unlike batted-ball metrics). Predictive for BABIP, triples,
   infield hit rate. Add to batter feature set.
2. **Multi-year traditional stat trends** — Delta features:
   `avg_1 - avg_2`, `slg_1 - slg_2`. Captures trajectory (improving vs declining
   players) which raw lag stats don't encode.

*Deferred to separate work:* Park factors (x-stats already park-neutral),
age curve interactions.

### Go/No-Go Gate

- **Per group:** Permutation importance ≥ 0.005 for at least one batter target,
  AND batter RMSE does not degrade on more than 1/6 targets.
- **Overall:** Batter evaluation RMSE improves on ≥ 3/6 targets vs Phase 2
  baseline.
- **No-go:** Remove groups that don't meet the per-group gate.

---

## Phase 4: Expand Training Data

**Rationale:** The model trains on 3 seasons (2022-2024). More data helps the
model learn regression patterns, and preseason mode is less sensitive to rule
changes than live mode (lag features already smooth out single-season effects).

### Work

1. Extend training to 2021-2024 (holdout: 2025). Skip 2020.
2. Optionally add an `era` indicator (0 = pre-pitch-clock, 1 = post).
3. Compare against the 3-season baseline.

### Go/No-Go Gate

- **Metric:** Holdout RMSE and 2025 evaluation RMSE.
- **Go:** Mean RMSE improves by ≥ 1%, no single target degrades > 5%.
- **No-go:** Try weighted training (recent seasons weighted more) or revert.

---

## Phase 5: Re-attempt Feature Pruning

**Rationale:** The initial pruning attempt failed because it was applied to the
original baseline. After phases 1-4 improve the model (better hyperparameters,
averaged features, new feature groups, more training data), the feature importance
landscape may look different. Features that were marginally useful before may
become clearly redundant, and the model may be robust enough to benefit from
pruning.

### Work

1. Re-run full ablation study on the post-Phase-4 model.
2. Apply the same pruning methodology: remove features with importance ≤ 0.
3. Compare holdout RMSE before and after pruning.

### Go/No-Go Gate

- **Metric:** Holdout RMSE after pruning vs Phase 4 baseline.
- **Go:** ≥ 4/8 holdout targets improve, no single target degrades > 5%.
- **No-go:** Accept the full feature set as final. The preseason model may
  genuinely need all features due to the noisier lagged signal.

---

## Summary

| Phase | Focus | Key Metric | Go Threshold |
|-------|-------|------------|--------------|
| 1 | Hyperparameter tuning | Holdout + eval RMSE | ≥ 2% mean improvement |
| 2 | Multi-year Statcast averaging | Batter holdout RMSE | ≥ 3/6 batter targets improve |
| 3 | New batter features | Batter RMSE + ablation | ≥ 0.005 importance/group |
| 4 | Expand training data | Holdout + eval RMSE | ≥ 1% mean improvement |
| 5 | Re-attempt feature pruning | Holdout RMSE | ≥ 4/8 targets improve |

Phases are ordered by expected impact-to-effort ratio. Each phase is independent —
a no-go on one phase does not block subsequent phases (except Phase 5, which
depends on earlier phases establishing a stronger baseline).

## Key Difference from Live Model

The preseason model is evaluated as a standard projection system — RMSE vs actuals
is the right metric, and it competes head-to-head with Marcel and Steamer. The
live model's true-talent evaluation suite (residual analysis, shrinkage quality,
etc.) does not apply here. The preseason model's goal is simply to predict future
stats as accurately as possible.
