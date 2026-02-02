# Projection Engine Evaluation Summary

Backtest of six Marcel pipeline variants against actual outcomes for 2021, 2022, 2023, and 2024 seasons. All values are 4-year averages.

**Filters:** min 200 PA (batting), min 50 IP (pitching). Top-N precision uses N=20.

## Batting Stat Accuracy

| Engine | HR RMSE | HR MAE | HR Corr | SB RMSE | SB MAE | SB Corr | OBP RMSE | OBP MAE | OBP Corr |
|--------|---------|--------|---------|---------|--------|---------|----------|---------|----------|
| marcel | 7.795 | 5.838 | 0.635 | 6.724 | 4.234 | 0.666 | 0.031 | 0.025 | 0.530 |
| marcel_park | 7.797 | 5.831 | 0.635 | 6.724 | 4.234 | 0.666 | 0.031 | 0.025 | 0.531 |
| marcel_statreg | 7.803 | 5.823 | 0.637 | **6.618** | **4.151** | **0.678** | 0.032 | 0.026 | 0.526 |
| marcel_plus | 7.802 | 5.814 | 0.637 | **6.618** | **4.151** | **0.678** | 0.032 | 0.026 | 0.527 |
| marcel_norm | 7.803 | 5.823 | 0.637 | **6.618** | **4.151** | **0.678** | 0.032 | 0.026 | 0.526 |
| marcel_full | 7.802 | 5.814 | 0.637 | **6.618** | **4.151** | **0.678** | 0.032 | 0.026 | 0.527 |

## Batting Rank Accuracy

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel | 0.558 | 0.513 |
| marcel_park | **0.562** | 0.513 |
| marcel_statreg | 0.558 | 0.513 |
| marcel_plus | 0.561 | 0.513 |
| marcel_norm | 0.558 | 0.513 |
| marcel_full | 0.561 | 0.513 |

## Pitching Stat Accuracy

| Engine | K RMSE | K MAE | K Corr | ERA RMSE | ERA MAE | ERA Corr | WHIP RMSE | WHIP MAE | WHIP Corr |
|--------|--------|-------|--------|----------|---------|----------|-----------|----------|-----------|
| marcel | 38.942 | 29.345 | 0.690 | 1.437 | 1.105 | 0.153 | 0.230 | 0.181 | 0.256 |
| marcel_park | 38.976 | 29.355 | 0.689 | 1.437 | 1.105 | 0.153 | 0.230 | 0.181 | 0.256 |
| marcel_statreg | 38.829 | 29.193 | 0.692 | 1.422 | 1.095 | 0.155 | 0.232 | 0.181 | 0.253 |
| marcel_plus | 38.859 | 29.204 | 0.691 | 1.422 | 1.095 | 0.155 | 0.232 | 0.181 | 0.253 |
| marcel_norm | **38.829** | **29.193** | **0.692** | **1.295** | **1.007** | **0.174** | **0.218** | **0.170** | **0.282** |
| marcel_full | 38.859 | 29.204 | 0.691 | 1.296 | 1.008 | 0.172 | **0.218** | 0.171 | 0.280 |

## Pitching Rank Accuracy

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel | 0.302 | 0.312 |
| marcel_park | 0.301 | 0.300 |
| marcel_statreg | 0.305 | 0.312 |
| marcel_plus | 0.303 | 0.300 |
| marcel_norm | **0.329** | **0.350** |
| marcel_full | 0.327 | 0.338 |

## Key Findings

### Pitcher BABIP/LOB% Normalization (marcel_norm)

The largest improvement of any pipeline change, targeting ERA and WHIP — previously the weakest projected metrics. Compared to marcel_statreg:

- **ERA RMSE dropped from 1.422 to 1.295 (-8.9%).** The single largest accuracy gain across all metrics and all pipeline changes. ERA correlation rose from 0.155 to 0.174 (+12.3%).
- **WHIP RMSE dropped from 0.232 to 0.218 (-6.0%).** WHIP correlation rose from 0.253 to 0.282 (+11.5%), driven by the same hit-rate regression that improves ERA.
- **Pitching rank accuracy jumped:** Spearman rho rose from 0.305 to 0.329 (+7.9%), top-20 precision from 0.312 to 0.350 (+12.2%). The normalization is particularly effective at separating good pitchers from bad by reducing BABIP/LOB noise.
- **No batting or K regression.** Batting metrics are identical to marcel_statreg. K projections are unchanged because the adjuster only modifies H and ER rates.

The adjuster regresses pitcher BABIP toward the league mean (weight 0.50) and LOB% toward a K%-adjusted expectation (weight 0.60), then recomputes hit and earned run rates. See `docs/evaluations/pitcher-normalization.md` for detailed analysis.

### Stat-Specific Regression (marcel_statreg)

The second-most impactful change. Compared to plain marcel:

- **SB projections improved substantially:** RMSE dropped from 6.724 to 6.618 (-1.6%), correlation rose from 0.666 to 0.678 (+1.8%). Driven by SB/CS getting a lower regression constant (600 PA vs. 1200) that trusts player speed signals more.
- **Pitching K projections improved:** RMSE dropped from 38.942 to 38.829, correlation rose from 0.690 to 0.692. K rate's low regression constant (30 outs) correctly identifies strikeout ability as a highly stable skill.
- **ERA/WHIP improved modestly:** ERA RMSE dropped from 1.437 to 1.422 (-1.0%), reflecting per-stat regression on noisy components like hits (200 outs) and earned runs (150 outs).

### Park Factor Adjustments (marcel_park)

Marginal impact, with mixed results:

- **Batting rank accuracy slightly improved:** Spearman rho rose from 0.558 to 0.562, suggesting park neutralization helps ordering even if counting stat RMSE is flat.
- **Counting stats essentially unchanged:** HR RMSE, SB metrics, and OBP metrics are within noise of plain marcel.
- **Pitching unaffected:** The park factor adjuster currently only maps batting-relevant stats (HR, 1B, 2B, 3B, BB, SO), so pitching rates pass through without meaningful adjustment.
- **Top-20 pitching precision dropped:** From 0.312 to 0.300, possibly due to slight rate perturbations propagating into ERA/WHIP for borderline cases.

The weak park factor signal likely reflects two issues: (1) the FanGraphs Guts page factors are already multi-year smoothed and regressed, and our provider applies additional regression on top, double-dampening the signal; (2) we only adjust rate stats — context-dependent stats like R and RBI that are most park-sensitive are not adjusted.

### Combined Engines (marcel_plus, marcel_full)

- **marcel_plus** (stat-specific regression + park factors) tracks marcel_statreg's gains with a small batting rank boost from park factors (Spearman 0.561).
- **marcel_full** (stat-specific regression + park factors + pitcher normalization) captures nearly all of marcel_norm's pitching gains while adding the park factor batting rank benefit. Pitching metrics are marginally below marcel_norm (top-20 precision 0.338 vs 0.350), suggesting park-adjusting rates before pitcher normalization introduces slight distortions.

## Recommendations

1. **Adopt marcel_norm as the new default.** It delivers the best pitching accuracy by a wide margin with no downside to batting projections.
2. **Use marcel_full when batting rank accuracy matters.** It adds a small park factor benefit (Spearman 0.561 vs 0.558) at the cost of slightly lower pitching top-20 precision.
3. **Tune pitcher normalization weights.** The current BABIP weight (0.50) and LOB weight (0.60) were set from priors. A grid search could yield further gains.
4. **Revisit park factor methodology.** Reduce or remove the second layer of regression, and consider adjusting R/RBI directly.
5. **Investigate per-pitcher BABIP skill.** Incorporating batted-ball profile data (ground-ball rate, infield-fly rate) could allow pitcher-specific BABIP targets instead of regressing all pitchers to the same league mean.
6. **Tune HR regression constant.** The current 500 PA may be too low; experiment with values between 500-800.

## Methodology

- **Evaluation years:** 2021, 2022, 2023, 2024 (4-year backtest)
- **Batting sample:** ~339 players/year with 200+ PA
- **Pitching sample:** ~316 players/year with 50+ IP
- **Metrics:** RMSE, MAE, Pearson correlation, Spearman rank correlation, Top-20 precision
- **Stat categories:** HR, SB, OBP (batting); K, ERA, WHIP (pitching)
