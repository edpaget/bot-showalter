# Data Profiling Tools

Enable an autonomous agent to understand what's in the raw statcast data before forming hypotheses about new features. Today, the closest tool is the ablation workflow, but it operates on already-curated features — there's no way to explore raw statcast columns or assess their potential signal before committing to a full feature implementation.

These tools answer the question: "What columns in `statcast_pitch` are worth turning into model features?"

## Status

| Phase | Status |
|-------|--------|
| 1 — Column profiler | done (2026-03-02) |
| 2 — Target correlation scanner | done (2026-03-02) |
| 3 — Temporal stability checker | in progress |

## Phase 1: Column profiler

Compute distribution statistics for any set of raw statcast columns, broken down by season and player type.

### Context

The `statcast_pitch` table has ~30 columns (release_speed, launch_speed, launch_angle, barrel, pfx_x, pfx_z, plate_x, plate_z, zone, spin rate, etc.), but the agent has no lightweight way to inspect their distributions. Knowing things like "barrel has 12% nulls in 2020 but 2% in other years" or "release_extension has a bimodal distribution" is critical before deciding whether a column is usable as a feature.

### Steps

1. Create `src/fantasy_baseball_manager/services/data_profiler.py` with a `profile_columns(columns, seasons, player_type)` function. For each column, compute: count, null count, null %, mean, median, std, min, max, p10, p25, p75, p90, skewness. Group results by season.
2. Define a `ColumnProfile` frozen dataclass in `src/fantasy_baseball_manager/domain/column_profile.py` holding the per-column, per-season statistics.
3. The profiler queries `statcast_pitch` directly, aggregating per batter_id or pitcher_id per season before computing stats. This gives player-season-level distributions (matching how features are actually used), not raw pitch-level distributions.
4. Add a `fbm profile columns <col1> <col2> ... --seasons 2021 2022 2023 2024 --player-type batter` CLI command that prints a summary table.
5. Support a `--all` flag that profiles every numeric column in `statcast_pitch` for a quick overview.
6. Write tests using synthetic statcast data verifying correct null %, mean, median, and per-season grouping.

### Acceptance criteria

- Profiler computes accurate distribution stats at the player-season level.
- Null percentages are reported per season (to catch data availability gaps like the 2020 shortened season).
- Output distinguishes batter-aggregated vs pitcher-aggregated profiles.
- `--all` flag works without specifying individual columns.

## Phase 2: Target correlation scanner

Given a candidate column (or aggregation), compute its correlation with each of the 13 model targets across seasons.

### Context

The first filter for any potential feature is: does it correlate with anything we're trying to predict? A raw statcast column that doesn't correlate with any of the 6 batter targets (avg, obp, slg, woba, iso, babip) or 7 pitcher targets (era, fip, k_per_9, bb_per_9, hr_per_9, babip, whip) is not worth pursuing. The agent needs a fast way to check this without building a full feature set.

### Steps

1. Add a `scan_target_correlations(column_spec, seasons, player_type)` function to the profiler service. `column_spec` is either a raw column name (e.g., `"launch_speed"`) or a SQL aggregation expression (e.g., `"AVG(launch_speed) WHERE barrel = 1"`).
2. For each season, aggregate the column to a player-season value, then join with actual batting/pitching stats to get target values. Compute Pearson and Spearman correlations between the candidate column and each target.
3. Return a `CorrelationScanResult` frozen dataclass: per-target Pearson r, Spearman rho, p-value, and sample size, both per-season and pooled across seasons.
4. Add a `fbm profile correlate "<expr>" --seasons 2021 2022 2023 2024 --player-type batter` CLI command that prints the correlation matrix.
5. Support scanning multiple columns at once and ranking them by average absolute correlation across targets.
6. Write tests with synthetic data where known linear relationships exist, verifying correct correlation values.

### Acceptance criteria

- Pearson and Spearman correlations are computed correctly against all relevant targets.
- Per-season and pooled-across-seasons results are both reported.
- SQL aggregation expressions work (e.g., conditional averages).
- Multi-column scan mode ranks columns by signal strength.
- Player-season values are joined to the correct actuals (same season).

## Phase 3: Temporal stability checker

Measure whether a feature-target relationship is consistent across seasons or a one-year fluke.

### Context

A column that correlates 0.6 with wOBA in 2023 but 0.05 in other years is noise, not signal. The agent needs to distinguish stable predictors from fluky ones. This is the difference between "barrel rate predicts SLG every year" and "spray angle happened to predict AVG once."

### Steps

1. Add a `check_temporal_stability(column_spec, target, seasons, player_type)` function that computes the correlation between the candidate column and a target for each season independently, then reports the cross-season consistency.
2. Report: per-season correlation, mean correlation, standard deviation of correlations, coefficient of variation (CV), and a stability score (e.g., "stable" if CV < 0.3, "unstable" if CV > 0.6, "moderate" otherwise).
3. Add a `fbm profile stability "<expr>" --target woba --seasons 2019 2021 2022 2023 2024 --player-type batter` CLI command.
4. Support a `--all-targets` flag that checks stability against every target and highlights which targets the feature is most consistently predictive of.
5. Write tests with synthetic data: one feature with a stable relationship across seasons, another with a one-year spike, verifying the stability classification.

### Acceptance criteria

- Per-season correlations are computed independently (not pooled).
- CV and stability classification correctly distinguish stable vs unstable features.
- The `--all-targets` flag produces a matrix of stability scores.
- 2020 short season can be excluded or flagged as an outlier season.

## Ordering

Phases are sequential: 1 → 2 → 3. Phase 1 is useful standalone for data exploration. Phase 2 depends on the profiler infrastructure from phase 1. Phase 3 extends the correlation scanner from phase 2 with temporal analysis. No dependencies on other roadmaps.
