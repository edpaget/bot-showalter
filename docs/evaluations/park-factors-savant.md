# Park Factor Evaluation: Savant Provider and Double-Dampening Fix

Backtest comparing park factor approaches against baselines. 3-year averages across 2022, 2023, 2024 seasons.

**Filters:** min 200 PA (batting), min 50 IP (pitching). Top-N precision uses N=20.

## Changes Tested

Two changes were applied simultaneously:

1. **Double-dampening fix:** `FanGraphsParkFactorProvider` defaults changed from `years_to_average=3, regression_weight=0.5` to `years_to_average=1, regression_weight=1.0`. FanGraphs Guts factors are already internally smoothed and regressed, so additional dampening was removed.

2. **Savant provider:** New `SavantParkFactorProvider` replacing the FanGraphs scraper. Scrapes Baseball Savant's Statcast park factors leaderboard, providing 12 stats (HR, 1B, 2B, 3B, BB, SO, runs, wOBA, OBP, hits, wOBAcon, xwOBAcon) with batter handedness split support. Uses 3-year rolling window.

Both `marcel_park` and `marcel_full` were tested with the Savant provider.

## Batting Stat Accuracy

| Engine | HR RMSE | HR MAE | HR Corr | SB RMSE | SB MAE | SB Corr | OBP RMSE | OBP MAE | OBP Corr |
|--------|---------|--------|---------|---------|--------|---------|----------|---------|----------|
| marcel | 7.315 | 5.582 | 0.650 | 7.218 | 4.549 | 0.651 | 0.031 | 0.025 | 0.541 |
| marcel_park (Savant) | **7.586** | **5.744** | **0.615** | 7.218 | 4.549 | 0.651 | **0.033** | **0.026** | **0.495** |
| marcel_norm | 7.365 | 5.594 | 0.652 | 7.130 | 4.477 | 0.665 | 0.032 | 0.025 | 0.538 |
| marcel_full (Savant) | **7.623** | **5.746** | **0.621** | 7.130 | 4.477 | 0.665 | **0.034** | **0.027** | **0.502** |

## Batting Rank Accuracy

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel | 0.557 | 0.533 |
| marcel_park (Savant) | **0.539** | **0.517** |
| marcel_norm | 0.557 | 0.533 |
| marcel_full (Savant) | **0.542** | **0.517** |

## Pitching Stat Accuracy

| Engine | K RMSE | K MAE | K Corr | ERA RMSE | ERA MAE | ERA Corr | WHIP RMSE | WHIP MAE | WHIP Corr |
|--------|--------|-------|--------|----------|---------|----------|-----------|----------|-----------|
| marcel | 36.321 | 26.954 | 0.701 | 1.423 | 1.086 | 0.147 | 0.229 | 0.178 | 0.246 |
| marcel_park (Savant) | **36.970** | **27.608** | **0.687** | 1.423 | 1.086 | 0.147 | 0.230 | 0.178 | 0.247 |
| marcel_norm | 36.273 | 26.899 | 0.703 | 1.277 | 0.984 | 0.167 | 0.216 | 0.167 | 0.270 |
| marcel_full (Savant) | **36.881** | **27.541** | **0.689** | 1.284 | 0.990 | 0.164 | 0.218 | 0.168 | 0.258 |

## Pitching Rank Accuracy

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel | 0.313 | 0.317 |
| marcel_park (Savant) | **0.308** | **0.300** |
| marcel_norm | 0.338 | 0.350 |
| marcel_full (Savant) | **0.329** | 0.350 |

## Key Findings

### Savant Park Factors Degrade Performance

The Savant provider made projections **worse** across the board compared to both baselines:

- **HR accuracy degraded significantly.** RMSE rose from 7.315 to 7.586 (+3.7%), correlation dropped from 0.650 to 0.615 (-5.4%). This is the largest single-metric regression observed in any pipeline change.
- **OBP correlation dropped sharply.** From 0.541 to 0.495 (-8.5%), the worst OBP performance of any engine variant tested.
- **Batting rank accuracy dropped.** Spearman rho fell from 0.557 to 0.539 (-3.2%), reversing the small improvement previously seen with FanGraphs factors.
- **Pitching K projections worsened.** RMSE rose from 36.321 to 36.970, correlation dropped from 0.701 to 0.687. Park factors are introducing noise into K rate projections.

### The Problem is Over-Adjustment, Not Signal Strength

The original FanGraphs implementation with double-dampening showed near-zero impact (within noise of baseline). Removing the dampening and switching to Savant's full-strength factors exposed that the rate-division approach itself introduces distortion:

1. **Extreme factors for rare events.** Savant reports triples factors as extreme as 1.80 (Coors) and 0.55 (Petco). Dividing a player's triples rate by 1.80 or 0.55 creates large distortions on small-count events that are inherently noisy.
2. **The adjustment conflates park and player.** Dividing rates by park factors assumes all of a player's stats are accumulated at their home park. In reality, players only play ~50% of games at home.
3. **Rate adjustment propagates through the pipeline.** Adjusting rates early in the pipeline (before rebaselining and aging) amplifies the distortion as downstream stages operate on already-perturbed values.
4. **More stats = more noise.** The Savant provider maps 12 stats vs. FanGraphs' 6, but many of the additional stats (wOBA, OBP, hits) are composites that compound per-component adjustment errors.

### marcel_full vs. marcel_norm Comparison

Even with the degradation, `marcel_full` still captures most of `marcel_norm`'s pitcher normalization gains (ERA RMSE 1.284 vs 1.277), confirming that pitcher BABIP/LOB% normalization is robust even when preceded by a suboptimal park factor adjustment. However, every metric is slightly worse than `marcel_norm`, confirming there is no benefit to including park factors in their current form.

## Decision

**Park factor pipelines reverted to FanGraphs provider** with the double-dampening fix (`years_to_average=1, regression_weight=1.0`). The `SavantParkFactorProvider` is retained in the codebase for future experimentation but is not wired into any active pipeline.

**`marcel_norm` remains the recommended default.** No park factor variant has outperformed it.

## Future Directions

Before park factors can improve projections, the following issues must be addressed:

1. **Half-game correction.** Players only play ~50% of games at their home park. The adjustment should use `(factor + 1.0) / 2.0` instead of the raw factor.
2. **Limit adjusted stats.** Only adjust high-signal stats (HR, possibly BB) where park effects are well-established. Drop low-signal adjustments (triples, SO, OBP).
3. **Pipeline ordering.** Consider adjusting after rebaselining rather than before, to avoid amplifying distortions through downstream stages.
4. **Pitching-specific factors.** Map HR park factor to pitching HR rate to capture the most impactful park effect on pitchers (see `docs/design-pitching-park-factors.md`).
5. **Moderate regression.** Even with pre-regressed data, a light regression (weight 0.8-0.9) may help smooth outlier parks.

## Methodology

- **Evaluation years:** 2022, 2023, 2024 (3-year backtest)
- **Batting sample:** ~336 players/year with 200+ PA
- **Pitching sample:** ~317 players/year with 50+ IP
- **Metrics:** RMSE, MAE, Pearson correlation, Spearman rank correlation, Top-20 precision
- **Stat categories:** HR, SB, OBP (batting); K, ERA, WHIP (pitching)
- **Park factor source:** Baseball Savant Statcast Park Factors, 3-year rolling, all bat sides
