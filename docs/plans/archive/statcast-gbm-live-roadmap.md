# Statcast-GBM Live Model Roadmap

**Model:** `statcast-gbm` (`StatcastGBMModel`)
**Created:** 2026-02-17
**Status:** Active
**Goal:** Improve the true-talent estimator through better evaluation tooling,
hyperparameter tuning, feature engineering, and expanded training data.

## Model Identity

`statcast-gbm` is a **true-talent estimator**, not a projection model. It uses
same-season Statcast features to estimate what a player's rate stats *should* be
given their underlying contact quality, plate discipline, and stuff metrics. The
residual (actual - estimate) reveals over/underperformance due to luck, sequencing,
or other transient factors.

**Use cases:**
- In-season buy-low/sell-high analysis (`fbm report talent-delta`)
- Ensemble component with Marcel (60/40 blend)
- Statcast-adjusted Marcel inputs (`statcast_augment = true`)

## Current Baseline (post-Phase 1 pruning)

Training: 2022-2024, holdout: 2025. Features pruned based on ablation study
(14 batter features removed, 20 pitcher features removed).

### Holdout RMSE

| Target | RMSE |
|--------|------|
| avg | 0.0311 |
| obp | 0.0336 |
| slg | 0.0595 |
| woba | 0.0358 |
| iso | 0.0357 |
| babip | 0.0683 |
| era | 6.153 |
| fip | 3.404 |
| k/9 | 1.981 |
| bb/9 | 2.710 |
| hr/9 | 5.264 |
| babip (pitcher) | 0.125 |
| whip | 0.850 |

### Completed Work

- **Feature pruning (Phase 1):** GO — 8/8 holdout targets improved. Removed 34
  features with zero or negative permutation importance.
- **Model split:** Extracted from shared `StatcastGBMModel` with preseason branching
  into dedicated `StatcastGBMModel` class with curated feature sets.
- **Phase 4a (Batted ball interactions):** NO-GO — failed importance gate. Code
  kept but excluded from curated columns.
- **Phase 4b (Sprint speed):** GO — permutation importance +0.0046, positive
  signal. Added to curated columns.

---

## Phase 1: True-Talent Evaluation Suite

**Priority:** High — this is the first phase to implement.

**Rationale:** Standard RMSE-vs-actuals is an incomplete metric for a true-talent
estimator. The model's value is not predicting the future but identifying signal vs
noise in current-season data. We need evaluation metrics that measure this directly.

### Work

Build a `TrueTalentEvaluator` (or extend `ProjectionEvaluator`) that computes the
following metrics, exposed via a new CLI command (e.g., `fbm evaluate-talent` or
`fbm report talent-quality`):

1. **Next-season predictive validity**
   - Compute model's true-talent estimates for season N (e.g., 2024)
   - Correlate those estimates with season N+1 actuals (2025)
   - Compare against: raw season-N stats → season N+1 actuals
   - **Success metric:** Model estimates have higher correlation with next-season
     actuals than raw stats do, for ≥ 5/8 target stats.
   - This is the most important test: it proves the model adds value beyond raw
     stats as a talent signal.

2. **Residual non-persistence**
   - Compute residuals (actual - estimate) for seasons N and N+1
   - Measure year-over-year correlation of residuals for returning players
   - **Success metric:** Residual correlation < 0.15 for ≥ 6/8 stats. Near-zero
     means the model correctly attributes transient noise to luck.

3. **Shrinkage quality**
   - Compare variance of model estimates vs variance of raw stats
   - Model estimates should have lower variance (shrunk toward the mean)
   - Compute ratio: `var(estimates) / var(raw_stats)` — should be < 1.0
   - **Success metric:** Shrinkage ratio < 0.9 for ≥ 4/8 stats while maintaining
     correlation ≥ 0.85 with raw stats.

4. **R-squared decomposition**
   - Compute within-season R² (how much variance the model explains)
   - Check residuals for systematic patterns by: player type, sample size (PA/IP
     buckets), park, division
   - **Success metric:** R² > 0.7 for ≥ 4/8 stats. No residual subgroup pattern
     with effect size > 0.02.

5. **Residual regression rate**
   - For players who over/underperformed in year N, measure how much they regress
     toward the model's estimate in year N+1
   - Compute: `1 - corr(residual_N, residual_N+1)` as the regression rate
   - **Success metric:** Regression rate > 0.80 (≥80% of residuals correct
     year-over-year) for ≥ 5/8 stats.

### Implementation approach

- Add a `TrueTalentQualityReport` dataclass to `domain/evaluation.py` to hold
  the metrics above.
- Build a service class that takes stored predictions for two consecutive seasons
  plus actuals, and computes all five metric families.
- Expose via `fbm report talent-quality --season 2024 --season 2025` (requires
  predictions and actuals for both seasons).
- Requires running `fbm predict statcast-gbm --season 2024` and
  `fbm predict statcast-gbm --season 2025` first.

### Go/No-Go Gate

This phase produces metrics, not model changes. The gate determines whether the
model is functioning correctly as a true-talent estimator:

- **Go (model is working):** Passes ≥ 3 of the 5 success metrics above.
- **Partial go:** Passes 1-2 metrics. Investigate which assumptions fail and
  whether model changes (later phases) could address them.
- **No-go (model is not a useful talent estimator):** Fails all metrics. The
  model should be repositioned as a same-season projection model rather than
  a true-talent estimator, and the evaluation doc updated accordingly.

---

## Phase 2: Hyperparameter Tuning

**Rationale:** The model uses scikit-learn defaults. Tuning could improve accuracy,
especially for the weaker batter targets.

### Work

1. Add a `tune` operation (or standalone script) that runs time-series
   cross-validation over a parameter grid:
   - `max_iter`: [100, 200, 500, 1000]
   - `max_depth`: [3, 5, 7, None]
   - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
   - `min_samples_leaf`: [5, 10, 20, 50]
   - `max_leaf_nodes`: [15, 31, 63, None]
2. Use the pruned (curated) feature sets.
3. Tune batter and pitcher sub-models separately — they may benefit from
   different hyperparameters.
4. Store best params in model config (fbm.toml or params file in artifacts/).

### Go/No-Go Gate

- **Metric:** Holdout RMSE and 2025 evaluation RMSE vs Phase 1 baseline.
- **Go:** Mean RMSE across all targets improves by ≥ 2%, with no single target
  degrading more than 5%.
- **No-go:** Keep defaults and move on. HistGradientBoosting defaults are
  generally strong.

---

## Phase 3: Remove Remaining Lag Stats

**Rationale:** Phase 1 pruning already removed most lag stats, but `so_1` (batter)
and `ip_1`, `era_1`, `fip_1` (pitcher) remain. For a true-talent estimator using
same-season data, lag stats are conceptually unnecessary — the model should rely
entirely on current-season Statcast signals. Removing them would make the model
purely same-season.

### Work

1. Create fully lag-free batter and pitcher feature sets.
2. Compare against the curated sets that still include the surviving lag stats.
3. Batter and pitcher decisions are independent.

### Go/No-Go Gate

- **Metric:** Holdout RMSE for live model only.
- **Go:** No target degrades more than 2%. Even flat results are a go because the
  simplification has conceptual value (pure same-season estimator).
- **No-go:** If any target degrades > 2%, restore lag stats for that player type.

---

## Phase 4: New Batter Features

**Status:** Partially complete. Sprint speed added; batted ball interactions
attempted but failed gate. Park factors and platoon splits deferred to later work.

**Rationale:** Batter features are the model's structural weakness. The top batter
feature (`avg_exit_velo`, +0.035 importance) is 8x weaker than the top pitcher
feature (`whiff_rate`, +0.270). New feature categories could unlock better batter
estimates.

### Completed

1. **Batted ball quality interactions (4a)** — NO-GO. Failed permutation
   importance gate. Code kept (BATTED_BALL_INTERACTIONS DerivedTransformFeature)
   but excluded from curated columns.
2. **Sprint speed (4b)** — GO. Permutation importance +0.0046. Added to live
   batter curated columns. New ingestion pipeline: `fbm ingest sprint-speed`.

### Deferred (later work)

3. **Park factors** — FanGraphs park factors for HR, H, 2B, 3B. Adjusts Statcast
   data for venue effects. Add as a context feature keyed to team.
4. **Platoon splits** — wOBA vs LHP / vs RHP from FanGraphs.

### Go/No-Go Gate

- **Per group:** Permutation importance ≥ 0.005 for at least one batter target,
  AND batter RMSE does not degrade on more than 1/6 targets.
- **Overall:** Batter evaluation RMSE improves on ≥ 3/6 targets vs Phase 3
  baseline.
- **No-go:** Remove groups that don't meet the per-group gate.

---

## Phase 5: Expand Training Data

**Status:** Complete (partial go).

**Rationale:** The model trains on 3 seasons (2022-2024). Adding 2021 (skipping
2020's 60-game season) provides more training examples, though pre-pitch-clock
data may add noise.

### Work

1. Extend training to 2021-2024 (holdout: 2025). Skip 2020. — **Done** (operational,
   pass `--season 2021` through `--season 2025` to CLI).
2. Optionally add an `era` indicator (0 = pre-pitch-clock, 1 = post). — **NO-GO**.
   `pitch_clock_era` showed +0.0000 permutation importance for both batter and
   pitcher models. The pitch-clock effect is already captured implicitly by the
   underlying Statcast features. Feature added and reverted.
3. Compare against the 3-season baseline. — Deferred (expanded training already
   in use).

### Go/No-Go Gate

- **Metric:** Holdout RMSE and 2025 evaluation RMSE.
- **Go:** Mean RMSE improves by ≥ 1%, no single target degrades > 5%.
- **No-go:** Try weighted training (recent seasons weighted more) or revert.

---

## Summary

| Phase | Focus | Status | Key Metric | Go Threshold |
|-------|-------|--------|------------|--------------|
| 1 | True-talent evaluation suite | Not started | 5 quality metrics | ≥ 3/5 metrics pass |
| 2 | Hyperparameter tuning | Not started | Holdout + eval RMSE | ≥ 2% mean improvement |
| 3 | Remove remaining lag stats | Not started | Holdout RMSE | No target degrades > 2% |
| 4 | New batter features | Partial | Batter RMSE + ablation | ≥ 0.005 importance/group |
| 5 | Expand training data | Complete (era indicator NO-GO) | Holdout + eval RMSE | ≥ 1% mean improvement |

Phase 4 status: 4a (interactions) NO-GO, 4b (sprint speed) GO, 4c (park factors)
and 4d (platoon splits) deferred.

Phases are ordered by priority. Phase 1 (evaluation suite) is prerequisite context
for later phases — it establishes whether the model is functioning as intended and
provides richer metrics for go/no-go decisions in subsequent phases.
