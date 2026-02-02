# Projection Engine Evaluation Summary

Backtest of four Marcel pipeline variants against actual outcomes for 2022, 2023, and 2024 seasons. All values are 3-year averages.

**Filters:** min 200 PA (batting), min 50 IP (pitching). Top-N precision uses N=20.

## Batting Stat Accuracy

| Engine | HR RMSE | HR MAE | HR Corr | SB RMSE | SB MAE | SB Corr | OBP RMSE | OBP MAE | OBP Corr |
|--------|---------|--------|---------|---------|--------|---------|----------|---------|----------|
| marcel | 7.315 | 5.582 | 0.650 | 7.218 | 4.549 | 0.651 | 0.031 | 0.025 | 0.541 |
| marcel_park | 7.320 | 5.573 | 0.650 | 7.218 | 4.549 | 0.651 | 0.031 | 0.025 | 0.540 |
| marcel_statreg | 7.365 | 5.594 | 0.652 | **7.130** | **4.477** | **0.665** | 0.032 | 0.025 | 0.538 |
| marcel_plus | 7.367 | 5.584 | 0.651 | **7.130** | **4.477** | **0.665** | 0.032 | 0.025 | 0.538 |

## Batting Rank Accuracy

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel | 0.557 | 0.533 |
| marcel_park | **0.559** | 0.533 |
| marcel_statreg | 0.557 | 0.533 |
| marcel_plus | **0.559** | 0.533 |

## Pitching Stat Accuracy

| Engine | K RMSE | K MAE | K Corr | ERA RMSE | ERA MAE | ERA Corr | WHIP RMSE | WHIP MAE | WHIP Corr |
|--------|--------|-------|--------|----------|---------|----------|-----------|----------|-----------|
| marcel | 36.321 | 26.954 | 0.701 | 1.423 | 1.086 | 0.147 | 0.229 | 0.178 | 0.246 |
| marcel_park | 36.351 | 26.970 | 0.700 | 1.423 | 1.086 | 0.147 | 0.229 | 0.178 | 0.246 |
| marcel_statreg | **36.273** | **26.899** | **0.703** | **1.408** | **1.076** | **0.149** | 0.230 | **0.177** | 0.243 |
| marcel_plus | 36.297 | 26.914 | 0.702 | **1.408** | **1.076** | **0.149** | 0.230 | **0.177** | 0.242 |

## Pitching Rank Accuracy

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel | 0.313 | **0.317** |
| marcel_park | 0.312 | 0.300 |
| marcel_statreg | **0.315** | **0.317** |
| marcel_plus | 0.314 | 0.300 |

## Key Findings

### Stat-Specific Regression (marcel_statreg)

The clearest improvement across the board. Compared to plain marcel:

- **SB projections improved substantially:** RMSE dropped from 7.218 to 7.130 (-1.2%), correlation rose from 0.651 to 0.665 (+2.1%). This is the largest single-metric gain, driven by SB/CS getting a lower regression constant (600 PA vs. 1200) that trusts player speed signals more.
- **Pitching K projections improved:** RMSE dropped from 36.321 to 36.273, correlation rose from 0.701 to 0.703. K rate's low regression constant (30 outs) correctly identifies strikeout ability as a highly stable skill.
- **ERA/WHIP improved modestly:** ERA RMSE dropped from 1.423 to 1.408 (-1.1%), reflecting the per-stat regression on noisy components like hits (200 outs) and earned runs (150 outs).
- **HR projections slightly worse:** RMSE rose from 7.315 to 7.365 (+0.7%). The higher regression for HR (500 PA vs. 1200) may be over-trusting small-sample HR rates. This warrants further tuning.

### Park Factor Adjustments (marcel_park)

Marginal impact, with mixed results:

- **Batting rank accuracy slightly improved:** Spearman rho rose from 0.557 to 0.559, suggesting park neutralization helps ordering even if counting stat RMSE is flat.
- **Counting stats essentially unchanged:** HR RMSE, SB metrics, and OBP metrics are within noise of plain marcel.
- **Pitching unaffected:** The park factor adjuster currently only maps batting-relevant stats (HR, 1B, 2B, 3B, BB, SO), so pitching rates pass through without meaningful adjustment.
- **Top-20 pitching precision dropped:** From 0.317 to 0.300, possibly due to slight rate perturbations propagating into ERA/WHIP for borderline cases.

The weak park factor signal likely reflects two issues: (1) the FanGraphs Guts page factors are already multi-year smoothed and regressed, and our provider applies additional regression on top, double-dampening the signal; (2) we only adjust rate stats — context-dependent stats like R and RBI that are most park-sensitive are not adjusted.

### Combined (marcel_plus)

Combines both improvements, generally tracking marcel_statreg's gains with marcel_park's small ranking boost. The best Spearman rho for batting (0.559) and competitive pitching metrics.

## Recommendations

1. **Adopt marcel_statreg as the new default.** It improves the most important metrics (SB, K, ERA) with minimal regression elsewhere.
2. **Revisit park factor methodology** before promoting marcel_park. Reduce or remove the second layer of regression, and consider adjusting R/RBI directly.
3. **Tune HR regression constant.** The current 500 PA may be too low; experiment with values between 500-800.
4. **Expand pitching park factors.** Map HR park factor to pitching HR rate to capture home-run-friendly parks affecting pitcher projections.

## Methodology

- **Evaluation years:** 2022, 2023, 2024 (3-year backtest)
- **Batting sample:** ~335 players/year with 200+ PA
- **Pitching sample:** ~317 players/year with 50+ IP
- **Metrics:** RMSE, MAE, Pearson correlation, Spearman rank correlation, Top-20 precision
- **Stat categories:** HR, SB, OBP (batting); K, ERA, WHIP (pitching)
