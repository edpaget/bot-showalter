# Projection Engine Evaluation Summary

Backtest of Marcel pipeline variants against actual outcomes. Tables in the first section show the original six-engine comparison (2021-2024 4-year averages). The Statcast section below shows the extended comparison including Statcast-blended engines (2022-2024 3-year averages).

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

Marginal impact with FanGraphs factors, actively harmful with Savant factors:

- **FanGraphs (dampened):** Batting rank accuracy slightly improved (Spearman rho 0.558 to 0.562), counting stats unchanged, pitching precision dropped (0.312 to 0.300). The double-dampening (our regression on top of FanGraphs' internal regression) suppressed the signal to near-zero.
- **Savant (full-strength):** Removing dampening and switching to Baseball Savant's Statcast factors **degraded every metric**. HR RMSE rose 3.7%, OBP correlation dropped 8.5%, batting Spearman rho dropped 3.2%. Extreme factors for rare events (triples factor 1.80 at Coors) and the lack of a half-game correction (players play ~50% away) cause over-adjustment that introduces noise.

The current rate-division approach has fundamental issues that stronger factors only expose. See `docs/evaluations/park-factors-savant.md` for the full Savant evaluation. Pipelines reverted to FanGraphs provider with dampening fix (`years_to_average=1, regression_weight=1.0`); no park factor variant outperforms `marcel_norm`.

### Combined Engines (marcel_plus, marcel_full)

- **marcel_plus** (stat-specific regression + park factors) tracks marcel_statreg's gains with a small batting rank boost from park factors (Spearman 0.561).
- **marcel_full** (stat-specific regression + park factors + pitcher normalization) captures nearly all of marcel_norm's pitching gains while adding the park factor batting rank benefit. Pitching metrics are marginally below marcel_norm (top-20 precision 0.338 vs 0.350), suggesting park-adjusting rates before pitcher normalization introduces slight distortions.

### Statcast Contact Quality Blending (marcel_full_statcast)

3-year backtest (2022, 2023, 2024) comparing Statcast-enhanced pipelines. All values are 3-year averages.

| Engine | HR RMSE | HR Corr | OBP RMSE | OBP Corr | Bat Spearman | Bat Top-20 |
|--------|---------|---------|----------|----------|-------------|------------|
| marcel_full | 7.304 | 0.658 | 0.031 | 0.535 | 0.581 | 0.533 |
| marcel_plus_statcast | 7.261 | 0.665 | 0.031 | 0.545 | 0.586 | 0.533 |
| marcel_full_statcast | 7.261 | 0.665 | 0.031 | 0.545 | 0.586 | 0.533 |
| marcel_full_statcast_babip | 7.261 | 0.665 | 0.031 | 0.544 | 0.587 | 0.533 |

Statcast contact quality blending (xBA, xSLG, barrel rate) improved HR projections:

- **HR RMSE dropped from 7.304 to 7.261 (-0.6%).** HR correlation rose from 0.658 to 0.665 (+1.1%). Barrel rate provides a direct measure of power quality that historical HR counts miss.
- **OBP correlation improved from 0.535 to 0.545 (+1.9%).** Hit-type decomposition from xBA/xSLG better estimates singles, doubles, and triples rates.
- **Batting rank accuracy improved:** Spearman rho rose from 0.581 to 0.586 (+0.9%).
- **Pitching metrics identical** to marcel_full (Statcast blending only affects batters).

### Batter BABIP Adjustment (marcel_full_statcast_babip)

The `BatterBabipAdjuster` derives an expected BABIP from Statcast xBA and barrel rate, then adjusts the batter's singles rate toward that target. Compared to marcel_full_statcast:

- **Essentially neutral at default weight (0.5).** Batting Spearman rho moved from 0.586 to 0.587 (within noise). OBP correlation dipped trivially from 0.545 to 0.544.
- The stage does not degrade any metric, so it is safe to include. The adjustment weight may need tuning via grid search — the interaction between `StatcastRateAdjuster` (which already blends hit types) and `BatterBabipAdjuster` (which further corrects singles) may require a lower weight to avoid double-counting.
- R, RBI, SB, and all pitching metrics are unaffected (the stage only modifies batter singles).

### Pitcher Statcast Blending (marcel_full_statcast_pitching)

The `PitcherStatcastAdjuster` blends Statcast xBA-against and xERA with Marcel pitcher rates for hits and earned runs, targeting ERA and WHIP — the two weakest projected stats. 3-year backtest (2022-2024):

| Engine | ERA RMSE | ERA Corr | WHIP RMSE | WHIP Corr | Pitch Spearman | Pitch Top-20 |
|--------|----------|----------|-----------|-----------|----------------|--------------|
| marcel_full_statcast_babip | 1.210 | 0.209 | 0.208 | 0.293 | 0.413 | 0.350 |
| marcel_full_statcast_pitching | 1.211 | **0.247** | 0.217 | 0.266 | **0.418** | **0.367** |

Compared to marcel_full_statcast_babip:

- **ERA correlation jumped from 0.209 to 0.247 (+18.2%).** The largest single-metric correlation gain since pitcher normalization. xERA captures pitch quality and contact management that historical ER counts miss. ERA RMSE was essentially unchanged (1.210 vs 1.211).
- **Pitching rank accuracy improved:** Spearman rho rose from 0.413 to 0.418 (+1.2%), top-20 precision from 0.350 to 0.367 (+4.9%). The stage improves separation between elite and replacement-level pitchers.
- **WHIP degraded:** WHIP RMSE rose from 0.208 to 0.217 (+4.3%), correlation dropped from 0.293 to 0.266 (-9.2%). The hit rate blend from xBA-against may over-correct WHIP components. The h rate and er rate are adjusted together with the same blend weight (0.30); decoupling the weights or reducing the h blend may improve WHIP without sacrificing ERA gains.
- **Batting metrics identical** to marcel_full_statcast_babip (pitcher Statcast blending only affects pitchers).
- **K, W, NSVH unaffected** — the adjuster only modifies h and er rates.

## Recommendations

### Current defaults

1. **Use marcel_full_statcast_pitching as the default.** It combines all proven improvements (stat-specific regression, park factors, pitcher normalization, batter Statcast contact quality, batter BABIP adjustment, pitcher Statcast blending) and delivers the best ERA correlation and pitching rank accuracy alongside strong batting metrics.
2. **Use marcel_full_statcast_babip when WHIP accuracy is prioritized over ERA.** The pitcher Statcast blending improves ERA correlation (+18.2%) and pitching rank accuracy but degrades WHIP (-9.2% correlation). If WHIP is more important in the league scoring, use the non-pitching-Statcast variant.
3. **Use marcel_norm when Statcast data is unavailable.** It delivers the best pitching accuracy of the non-Statcast variants.

### High impact (targeting weakest metrics)

4. **Tune pitcher Statcast blend weight and decouple h/er weights.** The current 0.30 blend weight for both h and er rates improved ERA correlation significantly but degraded WHIP. Decoupling the h blend weight (lower) from the er blend weight (keep or increase) may preserve ERA gains while recovering WHIP accuracy. Grid search over h_weight 0.10-0.30 and er_weight 0.20-0.50.
5. **Per-pitcher BABIP skill model.** Replace the flat league-mean BABIP regression target with pitcher-specific targets derived from batted-ball profile (GB%, IFFB%). Ground-ball pitchers sustain genuinely lower BABIPs. See `docs/per-pitcher-babip-skill.md`.
6. **Playing time model overhaul.** The current linear model (`0.5*PA_y1 + 0.1*PA_y2 + 200`) doesn't account for injury history, role changes, or prospect call-ups. All counting stats (R, RBI, SB, W, K) scale with playing time, so improvements here propagate everywhere. See `docs/playing-time-modeling.md`.

### Medium impact

7. **Stolen base rule change calibration.** Re-tune SB/CS regression constants and aging curves for the post-2023 environment (larger bases, pitch clock). See `docs/stolen-base-rule-adjustment.md`.
8. **Platoon/handedness splits.** Project vs-LHP and vs-RHP rates separately, then blend by expected matchup frequency. See `docs/platoon-splits.md`.
9. **Revisit park factor methodology.** The rate-division approach needs a half-game correction, stat filtering (HR/BB only), and pipeline reordering. See `docs/evaluations/park-factors-savant.md`.

### Lower impact / longer term

10. **Bayesian regression.** Replace flat regression-to-mean with per-player priors based on experience and Statcast skill changes. Veterans regress toward their career baseline; rookies regress toward the league mean. See `docs/bayesian-regression.md`.
11. **Data-fitted aging curves.** Fit the component aging parameters to actual cohort data using the delta method, rather than hand-set values. See `docs/improved-age-curves.md`.
12. **Minor league equivalencies.** Translate MiLB performance into MLB-equivalent rates for rookies with limited MLB history. See `docs/minor-league-equivalencies.md`.
13. **Grid search BatterBabipAdjuster weight.** The BABIP adjuster is neutral at weight 0.5. Search over 0.1-0.9 alongside the Statcast blend weight to find the optimal combination.

## Methodology

- **Evaluation years:** 2021, 2022, 2023, 2024 (4-year backtest)
- **Batting sample:** ~339 players/year with 200+ PA
- **Pitching sample:** ~316 players/year with 50+ IP
- **Metrics:** RMSE, MAE, Pearson correlation, Spearman rank correlation, Top-20 precision
- **Stat categories:** HR, SB, OBP, R, RBI (batting); K, W, ERA, WHIP, NSVH (pitching)
- **Statcast comparison:** 3-year backtest (2022, 2023, 2024) for engines requiring Statcast data
