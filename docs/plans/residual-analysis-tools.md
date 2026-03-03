# Residual Analysis Tools

Enable an autonomous agent to understand *where* the statcast GBM model fails, so it knows *what kind* of feature or adjustment might help. Raw RMSE numbers tell the agent the model is wrong; residual analysis tells it *why* and *for whom*. These tools turn model errors into actionable hypotheses.

## Status

| Phase | Status |
|-------|--------|
| 1 — Error decomposition | done (2026-03-03) |
| 2 — Feature gap detector | not started |
| 3 — Bias-by-cohort report | not started |

## Phase 1: Error decomposition

For each target, identify the players with the largest holdout residuals and cluster them by shared characteristics.

### Context

When the model badly mispredicts a player, there's usually a reason — the player changed their approach, came back from injury, aged out, or has a profile the model hasn't seen before. The agent needs to see these patterns without manually inspecting individual players. By surfacing the worst misses and their common traits, the agent can form hypotheses like "the model over-predicts SLG for aging power hitters" or "the model under-predicts K/9 for pitchers who added a new pitch."

### Steps

1. Create `src/fantasy_baseball_manager/services/residual_analyzer.py` with an `analyze_residuals(predictions, actuals, features, target, top_n)` function. It computes residuals (predicted - actual), sorts by absolute residual, and returns the top-N worst misses with their feature values and metadata.
2. Define a `ResidualReport` frozen dataclass: `target`, `top_misses` (list of `PlayerResidual` with player_id, predicted, actual, residual, feature_values), `over_predictions` (top-N where predicted > actual), `under_predictions` (top-N where actual > predicted).
3. Add summary statistics for the miss population: mean age, position distribution, mean PA/IP, and the features with the largest mean difference between the worst-miss group and the rest.
4. Add a `fbm residuals <model> --target slg --season 2024 --top 20 --player-type batter` CLI command that prints the worst misses with their feature context.
5. Support `--direction over` or `--direction under` to focus on one-sided errors.
6. Write tests with synthetic predictions/actuals verifying correct residual ranking and summary statistics.

### Acceptance criteria

- Top-N worst misses are correctly identified by absolute residual.
- Over-predictions and under-predictions are separated.
- Feature values are included for each miss, enabling pattern recognition.
- Summary statistics highlight what distinguishes the worst-miss group from the rest.

## Phase 2: Feature gap detector

For the worst-predicted players, compare their feature distributions to well-predicted players and flag the features with the largest distributional gap.

### Context

If the worst misses all have extreme values on a feature that isn't in the model, that's a strong signal for a new feature. If they have extreme values on a feature that *is* in the model, the model may need more capacity in that region (deeper trees, more interaction features). This tool automates the detective work of comparing distributions between well-predicted and poorly-predicted cohorts.

### Steps

1. Add a `detect_feature_gaps(predictions, actuals, features, target, miss_threshold)` function to the residual analyzer. Split players into "well-predicted" (absolute residual below median) and "poorly-predicted" (absolute residual above the miss_threshold percentile, e.g., top 20%).
2. For each feature, compute the Kolmogorov-Smirnov statistic and mean difference between the two groups. Rank features by KS statistic (largest distributional gap first).
3. Separately analyze features that are in the model vs. features available in the raw data but not in the model. The latter are the most actionable — they represent untapped signal.
4. Return a `FeatureGapResult` frozen dataclass: ranked list of `FeatureGap` (feature_name, ks_statistic, p_value, mean_well, mean_poor, in_model: bool).
5. Add a `fbm residuals gaps <model> --target slg --season 2024 --player-type batter [--include-raw]` CLI command. The `--include-raw` flag includes raw statcast columns not in the model.
6. Write tests with synthetic data where one unused feature perfectly separates well-predicted from poorly-predicted players.

### Acceptance criteria

- Features are correctly ranked by distributional gap (KS statistic).
- In-model and not-in-model features are distinguished.
- The `--include-raw` flag profiles raw statcast columns against the residual split.
- A feature with no distributional gap correctly gets a near-zero KS statistic.

## Phase 3: Bias-by-cohort report

Split holdout residuals by demographic cohorts (age, position, handedness, experience) and report systematic biases.

### Context

A model might have good overall RMSE but systematically over-predict young players' AVG and under-predict veteran pitchers' ERA. These cohort-level biases are invisible in aggregate metrics but directly actionable — they suggest adding cohort-specific features or applying calibration adjustments. The agent needs this view to prioritize which player segments to focus on.

### Steps

1. Add a `bias_by_cohort(predictions, actuals, player_metadata, target, dimension)` function to the residual analyzer. `dimension` is one of: `"age"` (buckets: 22-25, 26-29, 30-33, 34+), `"position"` (C, 1B, 2B, SS, 3B, OF, DH, SP, RP), `"handedness"` (L, R, S for batters; L, R for pitchers), `"experience"` (1-2 years, 3-5, 6-10, 11+).
2. For each cohort, compute: mean residual (bias direction), mean absolute residual (error magnitude), RMSE, sample size, and a significance indicator (is the bias statistically different from zero at p < 0.05?).
3. Return a `CohortBiasReport` frozen dataclass: dimension, list of `CohortBias` (cohort_label, n, mean_residual, mean_abs_residual, rmse, significant: bool).
4. Add a `fbm residuals cohort <model> --target era --season 2024 --dimension age --player-type pitcher` CLI command.
5. Support `--all-dimensions` to run all four dimensions at once and highlight the most biased cohorts across all dimensions.
6. Write tests with synthetic data where one cohort has a systematic bias and others are well-calibrated.

### Acceptance criteria

- Cohort splits are correct (age buckets, position groupings, etc.).
- Mean residual correctly captures bias direction (positive = over-prediction).
- Significance testing identifies cohorts with statistically meaningful bias.
- `--all-dimensions` surfaces the most biased cohorts across all groupings.

## Ordering

Phases are sequential: 1 → 2 → 3. Phase 1 provides the core residual computation and worst-miss identification. Phase 2 builds on phase 1 to compare feature distributions. Phase 3 provides a different slicing of the same residual data. No hard dependencies on other roadmaps, though the feature gap detector (phase 2) is most useful when the feature-candidate-factory is available to quickly test gap-filling features.
