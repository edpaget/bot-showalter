# Feature Ablation Improvement Roadmap

**Created:** 2026-02-18
**Status:** Phases 1–4 complete
**Goal:** Fix the feature ablation pipeline so pruning decisions are reliable.
Pruning attempts on both live and preseason models have produced mixed results —
the live model benefited but the preseason model degraded on re-attempt. The
root causes are methodological limitations in how permutation importance is
computed and how pruning decisions are made from it.

## Problem Statement

The current ablation pipeline (`compute_permutation_importance` in
`gbm_training.py`) has three structural problems:

### 1. Single-feature permutation misses correlated feature groups

Permutation importance shuffles one feature at a time. When features are
correlated (e.g., `barrel_pct` and `hard_hit_pct`, or `gb_pct` and `fb_pct`),
shuffling one has minimal effect because the correlated partner compensates.
Each individually appears unimportant, but removing the entire group causes
significant degradation.

In the Phase 5 preseason attempt, 7 individually "zero-importance" batter
features were removed simultaneously. The result: 10-16% RMSE degradation
across all batter targets. The features were redundant *individually* but
not *collectively*.

### 2. Ablation uses default hyperparameters, not tuned ones

The `ablate` operation in `_StatcastGBMBase` passes `config.model_params`
(which defaults to `{}`) directly to `fit_models`. But `train` routes
per-type params:

```python
# train():
batter_params = config.model_params.get("batter", config.model_params)
pitcher_params = config.model_params.get("pitcher", config.model_params)

# ablate():
bat_models = fit_models(bat_X_train, bat_y_train, config.model_params)  # always {}
```

The pitcher ablation computes importance under default params but the model
trains with tuned params (`learning_rate=0.01, max_depth=3, ...`). Feature
importance can differ substantially between model configurations — a feature
that's irrelevant to a deep tree may be critical to a shallow one.

### 3. Binary threshold (<=0) is too aggressive

The current approach prunes all features with importance <= 0. With noisy
importance estimates (5 repeats, moderate holdout size), features at the
boundary (+0.0000 to +0.0002) are within noise. Treating the threshold as
a hard cutoff leads to over-pruning.

## Phases

### Phase 1: Use tuned hyperparameters in ablation ✅

**Effort:** Small
**Risk:** Low

Fix the `ablate` method to route per-type params the same way `train` does.

**Changes:**
- `model.py`: In `_StatcastGBMBase.ablate()`, extract `batter_params` and
  `pitcher_params` the same way `train()` does, and pass them to `fit_models`.
- Tests: Verify ablation uses per-type params when provided.

**Validation:** Re-run ablation on preseason model, confirm pitcher importance
values shift compared to the default-param run.

### Phase 2: Group-aware importance (correlated feature detection) ✅

**Effort:** Medium
**Risk:** Medium

Instead of evaluating features in isolation, detect correlated groups and
evaluate them together.

**Approach:**
1. Before computing importance, compute a feature correlation matrix on the
   holdout set.
2. Cluster features with absolute correlation > threshold (e.g., 0.7) into
   groups using agglomerative clustering or connected components.
3. For each group, shuffle all features in the group simultaneously, then
   measure RMSE increase. This gives *group importance*.
4. A group is a pruning candidate only if its group importance <= 0.
5. Within a kept group, individual permutation importance can still identify
   the weakest member for optional single-feature pruning.

**Changes:**
- `gbm_training.py`: Add `compute_grouped_permutation_importance()` that
  accepts a correlation threshold, clusters features, and returns both
  group-level and feature-level importance.
- `model.py`: Wire the new function into `ablate()`, report both levels.
- CLI: Update ablation output to show groups and their importance.

**Validation:** Re-run on preseason model. Groups like
{`barrel_pct`, `hard_hit_pct`, `sweet_spot_pct`} should show meaningful
group importance even if individual importance is near zero.

### Phase 3: Increase statistical confidence of importance estimates ✅

**Effort:** Small
**Risk:** Low

Reduce noise in importance estimates so boundary decisions are more reliable.

**Approach:**
- Increase `n_repeats` from 5 to 20 (or make it configurable via CLI).
- Report standard error alongside mean importance.
- Use a conservative pruning rule: only prune if the upper bound of a 95%
  confidence interval (mean + 2*SE) is <= 0. This ensures we only prune
  features where we're confident they don't help.

**Changes:**
- `gbm_training.py`: Return per-feature mean and standard error from
  `compute_permutation_importance()`. Accept `n_repeats` from config.
- `model.py` / CLI: Pass through configurable repeat count, display
  confidence intervals.

**Validation:** Re-run ablation. Features that previously showed +0.0000
should now show either clearly positive (keep) or clearly zero (prune)
with tight confidence intervals.

### Phase 4: Forward selection validation ✅

**Effort:** Medium
**Risk:** Low

Add a complementary validation step: after identifying pruning candidates
via ablation, verify the pruned model doesn't degrade before committing.

**Approach:**
1. Run ablation to identify candidates (using improvements from Phases 1-3).
2. Train two models on the same data: full features vs. pruned features.
3. Compare holdout RMSE per-target. Apply the go/no-go gate (>= 4/8
   targets improve, no target degrades > 5%) automatically.
4. Report the comparison as part of the ablation output so the operator
   can make an informed decision before changing code.

This is essentially what we did manually in Phase 5 of the preseason
roadmap, but automated and run before any code changes.

**Changes:**
- `gbm_training.py` or new module: Add `validate_pruning()` that trains
  full and pruned models side-by-side and returns per-target deltas.
- `model.py`: Add a `prune` operation (or extend `ablate`) that runs
  ablation + validation in one pass.
- CLI: New command or flag, e.g., `fbm ablate --validate`.

**Validation:** Run on both live and preseason models. The live model
(where pruning previously succeeded) should show the same go signal.
The preseason model should show a clear no-go, matching our manual finding.

### Phase 5: Multi-holdout ablation

**Effort:** Medium
**Risk:** Low

Currently ablation uses a single holdout year (the last season). Importance
estimates from a single holdout can be noisy and year-specific.

**Approach:**
- Run ablation across multiple holdout years using temporal expanding CV
  (same splits as `tune`). Average importance across holdouts.
- Features must be consistently unimportant across holdouts to be pruned.

**Changes:**
- `gbm_training.py`: Add `compute_cv_permutation_importance()` that runs
  importance across multiple train/holdout splits and averages.
- `model.py`: Wire into ablation when multiple seasons are provided.

**Validation:** Compare single-holdout vs. multi-holdout importance
rankings. Multi-holdout should be more stable and produce more conservative
(safer) pruning decisions.

## Ordering and Dependencies

Phases 1-3 are independent and can be done in any order. Phase 4 depends
on Phases 1-3 being complete (it validates the improved ablation). Phase 5
is independent but benefits from Phase 3's confidence intervals.

Recommended order: **1 → 3 → 2 → 4 → 5**.

Phase 1 is a quick bug fix. Phase 3 is a small statistical improvement.
Phase 2 is the most impactful change. Phase 4 automates the validation
we've been doing manually. Phase 5 is an incremental robustness improvement.

## Success Criteria

After completing Phases 1-4, re-attempt preseason feature pruning. The
improved ablation should either:
- Identify a conservative pruning that passes the go/no-go gate, or
- Correctly indicate that the full feature set is already near-optimal
  (i.e., few or no features have reliably negative importance).

Either outcome is a success — the goal is *reliable decisions*, not
necessarily smaller feature sets.
