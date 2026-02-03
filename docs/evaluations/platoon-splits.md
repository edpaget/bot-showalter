# Platoon Splits Evaluation

Backtest of platoon-aware batting projections against `marcel_norm` baseline, using 2021-2024 seasons (4-year average). The platoon system computes batting rates separately for vs-LHP and vs-RHP matchups, applies 2x regression constants to account for smaller split samples, then blends by expected matchup frequency (72% vs RHP / 28% vs LHP).

**Filters:** min 200 PA (batting), min 50 IP (pitching). Top-N precision uses N=20.

## Motivation

Standard Marcel projections treat each batter's performance as a single aggregate line regardless of pitcher handedness. In reality, platoon splits are a well-documented skill: most batters hit meaningfully better against opposite-hand pitchers, and the magnitude of this advantage varies by player.

By projecting vs-LHP and vs-RHP performance separately and blending, we should capture individual platoon profiles that a single-rate approach averages away. This should improve accuracy for batters with strong platoon tendencies (e.g., left-handed batters who crush right-handed pitching but struggle against lefties).

The 2x regression constants reflect the reduced sample sizes in split data (~28% of PA vs LHP, ~72% vs RHP for a typical batter), which require heavier regression toward the league mean.

## Method

A `PlatoonRateComputer` replaces `StatSpecificRegressionRateComputer` for batting while delegating pitching unchanged. For each batter it:

1. Fetches vs-LHP and vs-RHP batting stats from FanGraphs split leaderboards for each historical year (3 years back)
2. Computes league-average rates from full-season team totals
3. For each split, applies Marcel-weighted regression using doubled regression constants (e.g., HR regression PA = 1000 instead of 500)
4. Blends: `rate = 0.72 * rate_vs_rhp + 0.28 * rate_vs_lhp`
5. Players missing from one split regress fully to league average for that split

The pipeline configuration is:

- **marcel_platoon:** `PlatoonRateComputer` + `RebaselineAdjuster` + `ComponentAgingAdjuster`

Note: `marcel_platoon` does not include `PitcherNormalizationAdjuster`, so pitching metrics are expected to be worse than `marcel_norm`. A future variant could combine both features.

## Results

### Batting Stat Accuracy (4-year avg, n=1356)

| Engine | HR RMSE | HR MAE | HR Corr | SB RMSE | SB MAE | SB Corr | OBP RMSE | OBP MAE | OBP Corr |
|--------|---------|--------|---------|---------|--------|---------|----------|---------|----------|
| marcel_norm | 7.789 | 5.779 | 0.642 | 6.671 | 4.056 | 0.691 | 0.032 | 0.025 | 0.518 |
| marcel_platoon | 7.792 | 5.807 | 0.642 | 6.984 | 4.277 | 0.669 | 0.031 | 0.025 | 0.487 |

### Batting Rank Accuracy (4-year avg, n=1356)

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel_norm | 0.575 | 0.488 |
| marcel_platoon | 0.578 | 0.488 |

### Pitching Stat Accuracy (4-year avg, n=1262)

| Engine | K RMSE | K MAE | K Corr | ERA RMSE | ERA MAE | ERA Corr | WHIP RMSE | WHIP MAE | WHIP Corr |
|--------|--------|-------|--------|----------|---------|----------|-----------|----------|-----------|
| marcel_norm | 39.267 | 29.747 | 0.691 | 1.229 | 0.960 | 0.219 | 0.210 | 0.164 | 0.307 |
| marcel_platoon | 39.267 | 29.747 | 0.691 | 1.407 | 1.088 | 0.173 | 0.227 | 0.178 | 0.266 |

### Pitching Rank Accuracy (4-year avg, n=1262)

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel_norm | 0.372 | 0.400 |
| marcel_platoon | 0.322 | 0.312 |

## Analysis

### Batting: Marginal Changes

The platoon system produces nearly identical batting accuracy to the baseline:

| Metric | norm | platoon | Change |
|--------|------|---------|--------|
| HR RMSE | 7.789 | 7.792 | +0.0% |
| HR Corr | 0.642 | 0.642 | +0.0% |
| SB RMSE | 6.671 | 6.984 | +4.7% |
| SB Corr | 0.691 | 0.669 | -3.2% |
| OBP RMSE | 0.032 | 0.031 | -3.1% |
| OBP Corr | 0.518 | 0.487 | -6.0% |
| Spearman rho | 0.575 | 0.578 | +0.5% |
| Top-20 precision | 0.488 | 0.488 | +0.0% |

HR projections are virtually unchanged. SB accuracy degraded slightly (+4.7% RMSE, -3.2% correlation), likely because stolen base rates have weak platoon signal — base-stealing is an overall speed skill, not matchup-dependent, so splitting the data introduces noise without additional signal. OBP correlation dropped 6.0%, suggesting the split-and-blend approach loses some information compared to the larger aggregate sample.

The one positive signal is Spearman rho improving marginally from 0.575 to 0.578, indicating the platoon system preserves overall ranking quality.

### Pitching: Expected Degradation

Pitching metrics are worse because `marcel_platoon` lacks `PitcherNormalizationAdjuster`. K projections are identical (confirming delegation works correctly), but ERA and WHIP metrics regress to pre-normalization levels. This is not a platoon-specific issue.

### Why Platoon Splits Didn't Help More

Several factors likely explain the neutral-to-negative batting results:

1. **Heavy regression washes out splits.** The 2x regression constants (e.g., 1000 PA for HR) relative to typical split sample sizes (~150 PA vs LHP, ~400 PA vs RHP per year) mean most players regress heavily toward the same league average in both splits, producing rates close to the aggregate projection.

2. **League averages aren't split-specific.** The current implementation uses full-season league averages as the regression target for both splits. In reality, league-wide OBP vs LHP differs from OBP vs RHP. Using split-specific league baselines would preserve the population-level platoon effect.

3. **SB and CS have weak platoon signal.** Stolen bases are primarily a function of runner speed and catcher arm, not pitcher handedness. Splitting this data just adds noise.

4. **Evaluation uses full-season actuals.** The evaluation compares projected totals against actual full-season stats, which already blend both splits. A platoon system would show its value more clearly in a matchup-level evaluation (e.g., predicting a batter's performance in a specific game against a known pitcher hand).

## Recommendations

1. **Do not adopt `marcel_platoon` as-is.** The batting improvements are negligible and come with OBP/SB degradation. The missing pitcher normalization makes it strictly worse overall.
2. **Consider a combined variant** (`marcel_platoon_norm`) that uses `PlatoonRateComputer` for batting with `PitcherNormalizationAdjuster` for pitching, to isolate the batting effect without pitching regression.
3. **Use split-specific league averages** as regression targets. The current approach regresses both splits toward the same full-season mean, losing the population-level platoon effect.
4. **Reduce regression intensity.** The 2x multiplier may be too aggressive. A grid search over regression multipliers (1.2x-2.0x) could find a better balance between noise reduction and signal preservation.
5. **Consider platoon data as a supplementary signal** rather than a replacement for aggregate rates — e.g., blend platoon-projected rates with aggregate-projected rates at some weight.
6. **Evaluate at the matchup level** to better capture the platoon system's value for in-season start/sit decisions, where knowing a batter's vs-LHP vs vs-RHP profile matters most.

## Methodology

- **Evaluation years:** 2021, 2022, 2023, 2024 (4-year backtest)
- **Batting sample:** ~339 players/year with 200+ PA
- **Pitching sample:** ~316 players/year with 50+ IP
- **Metrics:** RMSE, MAE, Pearson correlation, Spearman rank correlation, Top-20 precision
- **Stat categories:** HR, SB, OBP (batting); K, ERA, WHIP (pitching)
- **Split data source:** FanGraphs API leaderboard splits (month=13 for vs-LHP, month=14 for vs-RHP)
- **Platoon config:** pct_vs_rhp=0.72, pct_vs_lhp=0.28, regression PA = 2x standard Marcel constants
- **Caching:** SQLite-backed with 30-day TTL via `CachedSplitDataSource`
