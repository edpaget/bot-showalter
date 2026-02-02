# Pitcher BABIP/LOB% Normalization Evaluation

Backtest of pitcher normalization against `marcel_statreg` baseline, using 2021-2024 seasons (4-year average). The normalization regresses pitcher BABIP toward the league mean and LOB% toward a strikeout-adjusted expectation, then recomputes hit and earned run rates.

**Filters:** min 200 PA (batting), min 50 IP (pitching). Top-N precision uses N=20.

## Motivation

The prior evaluation (`evaluation-summary.md`) identified ERA as the weakest projected metric across all engines, with correlation at just 0.147-0.149. ERA is driven primarily by two components with low year-to-year stability for pitchers:

- **BABIP (Batting Average on Balls in Play):** League average ~.300. Pitcher-to-pitcher variation is mostly noise; year-to-year correlation is roughly 0.2-0.3. Pitchers with extreme observed BABIP are overwhelmingly likely to regress.
- **LOB% (Left on Base Percentage):** League average ~73%. Moderately influenced by strikeout rate (high-K pitchers strand more runners), but most variation is sequencing luck. Year-to-year correlation is similarly low.

By regressing these components toward stable expectations before computing ERA, we reduce the noise that raw observed rates carry forward into projections.

## Method

A `PitcherNormalizationAdjuster` is inserted into the pipeline after `RateComputer` and before `RebaselineAdjuster`. For each pitcher it:

1. Computes observed BABIP from per-out hit, home run, and strikeout rates
2. Regresses BABIP toward the league mean (weight 0.50) and recomputes the hit rate
3. Computes an expected LOB% from a strikeout-adjusted baseline (0.73 + 0.1 * (K% - league K%)), clamped to [0.65, 0.82]
4. Derives expected earned runs from the adjusted hit rate and expected LOB%, then blends with observed ER (weight 0.60 on expected)

Batters pass through unchanged. The adjuster reads league averages from pipeline metadata when available and falls back to configured defaults.

Two new presets were evaluated:

- **marcel_norm:** `StatSpecificRegressionRateComputer` + `PitcherNormalizationAdjuster` + `RebaselineAdjuster` + `MarcelAgingAdjuster`
- **marcel_full:** Same as marcel_norm but with `ParkFactorAdjuster` added before pitcher normalization

## Results

### Batting Stat Accuracy (n=1356)

| Engine | HR RMSE | HR MAE | HR Corr | SB RMSE | SB MAE | SB Corr | OBP RMSE | OBP MAE | OBP Corr |
|--------|---------|--------|---------|---------|--------|---------|----------|---------|----------|
| marcel_statreg | 7.803 | 5.823 | 0.637 | 6.618 | 4.151 | 0.678 | 0.032 | 0.026 | 0.526 |
| marcel_norm | 7.803 | 5.823 | 0.637 | 6.618 | 4.151 | 0.678 | 0.032 | 0.026 | 0.526 |
| marcel_full | 7.802 | 5.814 | 0.637 | 6.618 | 4.151 | 0.678 | 0.032 | 0.026 | 0.527 |

Batting metrics are identical for marcel_statreg and marcel_norm (expected, since batters pass through unchanged). marcel_full shows negligible differences from park factor adjustments.

### Batting Rank Accuracy (n=1356)

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel_statreg | 0.558 | 0.513 |
| marcel_norm | 0.558 | 0.513 |
| marcel_full | 0.561 | 0.513 |

### Pitching Stat Accuracy (n=1262)

| Engine | K RMSE | K MAE | K Corr | ERA RMSE | ERA MAE | ERA Corr | WHIP RMSE | WHIP MAE | WHIP Corr |
|--------|--------|-------|--------|----------|---------|----------|-----------|----------|-----------|
| marcel_statreg | 38.829 | 29.193 | 0.692 | 1.422 | 1.095 | 0.155 | 0.232 | 0.181 | 0.253 |
| marcel_norm | 38.829 | 29.193 | 0.692 | **1.295** | **1.007** | **0.174** | **0.218** | **0.170** | **0.282** |
| marcel_full | 38.859 | 29.204 | 0.691 | **1.296** | **1.008** | 0.172 | **0.218** | 0.171 | 0.280 |

### Pitching Rank Accuracy (n=1262)

| Engine | Spearman rho | Top-20 Precision |
|--------|-------------|-----------------|
| marcel_statreg | 0.305 | 0.312 |
| marcel_norm | **0.329** | **0.350** |
| marcel_full | 0.327 | 0.338 |

## Analysis

### Pitcher Normalization (marcel_norm vs marcel_statreg)

Every pitching metric improved, with no regression in batting or strikeout projections:

| Metric | statreg | norm | Change |
|--------|---------|------|--------|
| ERA RMSE | 1.422 | 1.295 | **-8.9%** |
| ERA MAE | 1.095 | 1.007 | **-8.0%** |
| ERA Corr | 0.155 | 0.174 | **+12.3%** |
| WHIP RMSE | 0.232 | 0.218 | **-6.0%** |
| WHIP MAE | 0.181 | 0.170 | **-6.1%** |
| WHIP Corr | 0.253 | 0.282 | **+11.5%** |
| Spearman rho | 0.305 | 0.329 | **+7.9%** |
| Top-20 precision | 0.312 | 0.350 | **+12.2%** |

ERA RMSE dropped from 1.422 to 1.295, the single largest improvement to any metric across all pipeline changes to date. ERA correlation rose from 0.155 to 0.174 — still the weakest individual stat, but a meaningful step. WHIP benefited from the same hit-rate regression, with correlation jumping from 0.253 to 0.282.

The pitching rank metrics saw the strongest relative gains: Spearman rho rose 7.9% and top-20 precision rose 12.2%, indicating the normalization is particularly effective at separating good pitchers from bad by reducing noise in who gets "lucky" BABIP/LOB seasons.

K projections were unaffected, confirming the adjuster correctly leaves strikeout rates untouched.

### Combined with Park Factors (marcel_full vs marcel_norm)

Adding park factors on top of pitcher normalization produced mixed results:

- Batting rank accuracy improved slightly (Spearman rho 0.558 to 0.561)
- Pitching metrics were marginally worse than marcel_norm alone (ERA corr 0.174 vs 0.172, top-20 precision 0.350 vs 0.338)

This is consistent with the earlier finding that park factor adjustments have minimal impact on pitching. The small degradation suggests that park-adjusting rates before pitcher normalization may introduce slight distortions. The ordering of these two adjusters could be revisited.

## Recommendations

1. **Adopt marcel_norm as the new default engine.** It delivers the largest accuracy improvement of any pipeline change, with no downsides to batting or strikeout projections.
2. **Consider marcel_full for batting-heavy contexts** where the small park factor benefit to batting rank accuracy matters, but prefer marcel_norm when pitching accuracy is the priority.
3. **Tune regression weights.** The current BABIP weight (0.50) and LOB weight (0.60) were set from priors about year-to-year stability. A grid search over these parameters could yield further gains.
4. **Investigate per-pitcher BABIP skill.** Some pitchers (high ground-ball rates, extreme infield-fly rates) have persistent BABIP deviations. Incorporating batted-ball profile data could allow pitcher-specific BABIP targets instead of regressing all pitchers to the same league mean.

## Methodology

- **Evaluation years:** 2021, 2022, 2023, 2024 (4-year backtest)
- **Batting sample:** ~339 players/year with 200+ PA
- **Pitching sample:** ~316 players/year with 50+ IP
- **Metrics:** RMSE, MAE, Pearson correlation, Spearman rank correlation, Top-20 precision
- **Stat categories:** HR, SB, OBP (batting); K, ERA, WHIP (pitching)
- **Default config:** league_babip=0.300, babip_regression_weight=0.50, lob_baseline=0.73, lob_k_sensitivity=0.10, league_k_pct=0.22, lob_regression_weight=0.60, min_lob=0.65, max_lob=0.82
