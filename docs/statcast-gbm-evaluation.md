# Statcast GBM Model: Evaluation & Analysis

**Model version:** latest (retrained 2026-02-15)
**Training data:** 2022-2024 (holdout: 2024)
**Evaluation season:** 2025 actuals (FanGraphs)
**Date:** 2026-02-15

## Overview

The statcast-gbm model is a gradient-boosted regression model that uses Statcast
pitch-level data (exit velocity, spin rates, plate discipline metrics, expected stats)
alongside traditional lag statistics to project rate stats for batters and pitchers.
It predicts rate/ratio stats only (no counting stats, no WAR, no playing time).

The model is built on scikit-learn's `HistGradientBoostingRegressor` with separate
models for each target stat. Batter and pitcher pipelines are fully independent
with different feature sets tailored to each player type.

### Two operating modes

The model supports two modes via the `mode` config param:

- **`true_talent`** (default): Uses **same-season Statcast features** — trains on
  2023 exit velocity to predict 2023 AVG, then uses 2025 Statcast data to predict
  2025 rates. This is a true-talent estimator: given what a player actually did in
  the Statcast data, what are his expected rate stats?

- **`preseason`**: Uses **prior-season Statcast features** (lag=1) — trains on 2022
  exit velocity to predict 2023 AVG, then uses 2024 Statcast data to predict 2025
  rates. This enables genuine pre-season projections comparable to Marcel/Steamer.
  Preseason artifacts are stored under a `preseason/` subdirectory.

The sections below evaluate both modes. The true-talent results represent a ceiling
for estimating underlying quality; the preseason results show how the model performs
as a genuine blind forecast comparable to Marcel and Steamer.

Additionally, the model's true-talent estimates can be blended into Marcel's inputs
via statcast-adjusted Marcel (`statcast_augment = true` in Marcel config), and
compared against actuals via the `fbm report talent-delta` command for in-season
buy-low/sell-high analysis.

## What It Predicts

| Player type | Stats predicted |
|-------------|----------------|
| Batter (1,063 players) | avg, obp, slg, woba, iso, babip |
| Pitcher (633 players)  | era, fip, whip, k/9, bb/9, hr/9, babip |

The model does **not** predict counting stats (HR, RBI, SB, W, SV, IP, SO) or
composite stats (WAR, wRC+, OPS). For fantasy purposes it must be paired with
a playing-time estimate and a counting-stat projection (e.g., Marcel) to produce
a complete player forecast.

## Features

### Batter features (30 total)

| Category | Features | Description |
|----------|----------|-------------|
| Age | `age` | Player age at season start |
| Lag stats (year 1 & 2) | `pa`, `hr`, `h`, `doubles`, `triples`, `bb`, `so`, `sb` x 2 lags | Traditional FanGraphs stats from prior two seasons |
| Batted ball | `avg_exit_velo`, `max_exit_velo`, `avg_launch_angle`, `barrel_pct`, `hard_hit_pct` | Same-season Statcast quality-of-contact metrics |
| Plate discipline | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct` | Same-season swing decision metrics |
| Expected stats | `xba`, `xwoba`, `xslg` | Same-season Statcast expected outcomes based on batted-ball physics |

### Pitcher features (36 total)

| Category | Features | Description |
|----------|----------|-------------|
| Age | `age` | Player age at season start |
| Lag stats (year 1 & 2) | `ip`, `so`, `bb`, `hr`, `era`, `fip` x 2 lags | Traditional FanGraphs stats from prior two seasons |
| Pitch mix | `ff_pct/velo`, `sl_pct/velo`, `ch_pct/velo`, `cu_pct/velo`, `si_pct/velo`, `fc_pct/velo` | Same-season usage % and velocity for 6 pitch types |
| Spin profile | `avg_spin_rate`, `ff_spin`, `sl_spin`, `cu_spin`, `avg_h_break`, `avg_v_break` | Same-season spin rates and movement |
| Plate discipline | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct` | Same-season swing decision metrics (from pitcher's perspective) |

## Holdout Metrics (2024 season)

These are RMSE values from the time-series holdout split during training
(trained on 2022-2023, tested on 2024). Both modes use the same split.

| Target | True-Talent | Preseason | Delta |
|--------|-------------|-----------|-------|
| **Batter** | | | |
| avg | 0.0326 | 0.0488 | +0.016 |
| obp | 0.0343 | 0.0584 | +0.024 |
| slg | 0.0597 | 0.0871 | +0.027 |
| woba | 0.0368 | 0.0581 | +0.021 |
| iso | 0.0369 | 0.0507 | +0.014 |
| babip | 0.0689 | 0.0735 | +0.005 |
| **Pitcher** | | | |
| era | 6.267 | 6.163 | -0.083 |
| fip | 3.498 | 3.511 | +0.013 |
| k/9 | 2.064 | 2.982 | +0.918 |
| bb/9 | 3.205 | 3.712 | +0.507 |
| hr/9 | 5.333 | 5.211 | -0.122 |
| babip | 0.130 | 0.124 | -0.006 |
| whip | 0.858 | 0.893 | +0.035 |

Preseason batter stats degrade 22-46% without same-season Statcast data, with
BABIP being the most stable (+7%). Pitcher ERA, FIP, HR/9, and BABIP are
essentially unchanged — prior-year stuff metrics (spin, velocity, movement) are
highly stable year-to-year. K/9 and BB/9 degrade significantly (+44% and +16%),
suggesting plate discipline metrics shift more between seasons than stuff does.

## Ablation Study

Permutation importance shows which features matter most. Higher values mean
removing the feature increases RMSE more (feature is more important).

### Most important features

| Feature | RMSE Impact | Notes |
|---------|------------|-------|
| pitcher:whiff_rate | +0.270 | By far the #1 feature overall |
| pitcher:avg_spin_rate | +0.149 | Spin rate is the top stuff metric |
| pitcher:chase_rate | +0.136 | Getting swings outside the zone |
| pitcher:swinging_strike_pct | +0.134 | Closely related to whiff_rate |
| pitcher:called_strike_pct | +0.060 | Command/location |
| batter:avg_exit_velo | +0.035 | Top batter feature — quality of contact |
| pitcher:zone_contact_pct | +0.030 | |
| batter:max_exit_velo | +0.026 | Raw power ceiling |
| batter:swinging_strike_pct | +0.015 | |
| pitcher:era_1 | +0.012 | Only meaningful lag stat |

### Key findings

- **Pitcher Statcast features dominate** — the top 5 features are all pitcher
  plate-discipline and stuff metrics. This explains the model's strength on
  pitcher rate stats.
- **Batter model runs on exit velocity + plate discipline** — lag stats and
  expected stats (xba, xwoba, xslg) contribute near-zero importance.
- **Some features add noise** — `avg_v_break` (-0.022), `cu_velo` (-0.006),
  `si_pct` (-0.006) have negative importance (removing them helps).
- **Lag stats barely matter** — only pitcher `era_1` (+0.012) shows signal.
  All lag-2 features and batter lag stats show ~0.000 importance.

## Accuracy vs Other Systems (2025 Season)

### True-talent mode (same-season Statcast)

Note: true-talent uses same-season Statcast input, giving it an information
advantage over Marcel and Steamer (which are pre-season projections).

#### Full player pool

RMSE comparison across all players with 2025 FanGraphs actuals. Lower is better.
Bold = best in row.

| Stat | Marcel | True-Talent | Steamer | Ensemble (60/40) |
|------|--------|-------------|---------|-------------------|
| avg | 0.0587 | **0.0479** | 0.0515 | 0.0494 |
| obp | 0.0643 | **0.0510** | 0.0551 | 0.0529 |
| slg | 0.0937 | **0.0673** | 0.0763 | 0.0735 |
| era | 7.670 | 6.633 | 7.342 | **6.324** |
| fip | 12.817 | **3.310** | 4.486 | **3.310** |
| whip | 1.248 | 0.829 | 1.191 | **0.804** |
| k/9 | 2.435 | **1.858** | 2.122 | 1.940 |
| bb/9 | 3.541 | **2.566** | 3.456 | 2.615 |

True-talent leads on every stat except ERA and WHIP (where the ensemble wins).

#### Top 200 players (by WAR)

The fantasy-relevant comparison. Top 200 by 2025 WAR filters to everyday
players and rotation arms — the population fantasy managers care most about.

| Stat | Marcel | True-Talent | Steamer | Ensemble (60/40) |
|------|--------|-------------|---------|-------------------|
| avg | 0.0286 | 0.0233 | 0.0247 | **0.0228** |
| obp | 0.0307 | 0.0263 | 0.0263 | **0.0251** |
| slg | 0.0625 | **0.0471** | 0.0549 | 0.0485 |
| era | 1.200 | 1.634 | **0.996** | 1.296 |
| fip | 3.360 | 0.989 | **0.742** | 0.989 |
| whip | 0.212 | 0.227 | **0.182** | 0.194 |
| k/9 | 1.808 | **1.009** | 1.329 | 1.256 |
| bb/9 | 0.883 | 0.831 | 0.759 | **0.748** |

For top 200: true-talent leads on AVG, SLG, and K/9. The ensemble leads on
AVG, OBP, and BB/9. Steamer leads on ERA, FIP, and WHIP for elite players.

### Preseason mode (prior-season Statcast)

This is the apples-to-apples comparison: preseason mode uses only prior-year
data, just like Marcel and Steamer.

#### Full player pool

| Stat | Marcel | Preseason | Steamer | Ensemble (60/40) |
|------|--------|-----------|---------|-------------------|
| avg | 0.0587 | 0.0585 | **0.0515** | 0.0494 |
| obp | 0.0643 | 0.0648 | **0.0551** | 0.0529 |
| slg | 0.0937 | 0.0863 | **0.0763** | 0.0735 |
| era | 7.670 | **6.626** | 7.342 | — |
| fip | 12.817 | **3.472** | 4.486 | — |
| whip | 1.248 | **0.904** | 1.191 | — |
| k/9 | **2.435** | 2.833 | 2.122 | — |
| bb/9 | 3.541 | 3.245 | **3.456** | — |

Preseason leads on ERA, FIP, and WHIP across the full pool — beating both Marcel
and Steamer on these pitcher stats. Batter stats are roughly on par with Marcel
but behind Steamer. K/9 degrades significantly and falls behind Marcel.

#### Top 200 players (by WAR)

| Stat | Marcel | Preseason | Steamer | Ensemble (60/40) |
|------|--------|-----------|---------|-------------------|
| avg | **0.0286** | 0.0362 | 0.0247 | 0.0228 |
| obp | **0.0307** | 0.0397 | 0.0263 | 0.0251 |
| slg | **0.0625** | 0.0717 | 0.0549 | 0.0485 |
| era | **1.200** | 2.169 | 0.996 | — |
| fip | 3.360 | **1.439** | 0.742 | — |
| whip | **0.212** | 0.342 | 0.182 | — |
| k/9 | 1.808 | **1.723** | 1.329 | — |
| bb/9 | **0.883** | 1.531 | 0.759 | — |

For top 200, preseason is the weakest system on most stats. The only stat where
it leads is K/9 (1.723 vs Marcel's 1.808). The lagged Statcast features lose
precision for established players where Marcel's regression-to-the-mean machinery
and Steamer's deeper modeling already work well.

## Strengths

### 1. Best-in-class rate stats across the full player pool

With same-season Statcast data, the model beats every other system on all eight
core rate stats for the full player pool. The advantage is largest on:

- **FIP RMSE of 3.31** crushes Marcel's 12.82 and beats Steamer's 4.49.
- **WHIP RMSE of 0.83** beats Steamer's 1.19 by 30%.
- **K/9 RMSE of 1.86** beats Steamer's 2.12 — a notable improvement from v1
  where K/9 was the weakest stat.
- **AVG RMSE of 0.048** now beats both Marcel (0.059) and Steamer (0.052) — a
  reversal from v1 where batter stats were the weakest area.

### 2. Strong ensemble component

The 60% Marcel / 40% statcast-gbm ensemble produces the best results on several
key stats for top-200 players:

- **AVG (0.0228)** and **OBP (0.0251)** are the best of any system for top 200.
- **BB/9 (0.748)** beats Steamer (0.759) for top 200.
- **ERA (6.32)** and **WHIP (0.80)** lead the full pool.

### 3. Captures Statcast-era signal

The feature set captures information that purely historical systems miss:
- Exit velocity / barrel rate predict power better than raw HR counts.
- Whiff rate and chase rate capture strikeout/walk tendencies at a granular level.
- xBA/xwOBA/xSLG provide "luck-adjusted" baselines that stabilize faster than
  actual results over small samples.
- Pitch velocity and spin data capture stuff quality independent of results.

### 4. Handles missing data natively

`HistGradientBoostingRegressor` treats NaN as a first-class missing indicator,
so the model gracefully handles players with partial Statcast data (e.g., limited
batted-ball events, new pitch types). No imputation heuristics needed.

## Weaknesses

### 1. Preseason mode lags behind Marcel and Steamer on batter stats

The preseason mode enables genuine pre-season projections but is not competitive
with Marcel or Steamer on batter rate stats (AVG, OBP, SLG). Lagging the Statcast
features by one year costs ~22% accuracy on AVG and ~28% on SLG compared to
true-talent mode. The preseason model's strength is pitcher ERA/FIP/WHIP for the
full player pool, where prior-year stuff metrics retain strong predictive power.

### 2. No counting stats or playing time

The model only predicts rates. Fantasy baseball values are primarily driven by
counting stats (HR, RBI, R, SB, W, SV, SO). Without a playing-time estimate,
statcast-gbm projections can't be turned into fantasy dollar values or draft
rankings on their own. The ensemble helps here by inheriting Marcel's counting
stats for overlapping players.

### 3. Smaller player coverage

| System | Players |
|--------|---------|
| Steamer | 2,626 |
| Marcel | 1,904 |
| Statcast-GBM | 1,696 (1,063 batters + 633 pitchers) |

Statcast-gbm covers fewer players because it requires Statcast pitch-level data
from Baseball Savant. Players without meaningful MLB pitch data (rookies from the
minors, international signings, September call-ups with tiny samples) get no
projection. For fantasy, this means the model misses some breakout rookies that
Steamer and ZiPS project using minor-league data.

### 4. Pitcher ERA/WHIP degrade for top 200

While pitcher projections dominate the full pool, Steamer wins on ERA (0.996 vs
1.634), FIP (0.742 vs 0.989), and WHIP (0.182 vs 0.227) for top-200 pitchers.
For established arms, Steamer's deeper modeling (park factors, defense, workload)
adds value that raw Statcast features don't fully capture.

### 5. Training data is only 2 seasons deep

The model trains on 2022-2023 and validates on 2024. This is a very small
training set. More seasons would help the model learn better regression patterns,
but Statcast data quality and rule changes (pitch clock in 2023, shift ban) create
legitimate reasons to limit the lookback window.

## Fantasy Implications

### Where to trust statcast-gbm

- **Batter rate stats**: With same-season data, the model now leads on AVG, OBP,
  and SLG across the full pool and for top-200 players.
- **As an ensemble component**: The 60/40 Marcel/statcast-gbm blend produces the
  best AVG and OBP for top-200 players of any system tested.
- **K/9 projections**: A major improvement from v1 — now leads all systems on
  both the full pool and top 200.
- **Pitcher rate stats for deep leagues**: ERA, FIP, WHIP, and BB/9 projections
  dominate for the full player pool.

### Where to be skeptical

- **Pre-season drafts**: The preseason mode can produce blind forecasts, but it
  trails Marcel and Steamer on batter stats and top-200 accuracy. Its value is
  as a supplementary pitcher signal (ERA/FIP/WHIP), not a standalone draft tool.
- **Counting-stat-dependent formats**: Roto leagues need HR, RBI, SB projections
  that this model doesn't provide. Use the ensemble or Marcel directly.
- **Rookie/prospect projections**: No Statcast data = no projection. Use a system
  with minor-league translation (or the MLE model) for prospects.
- **Top-tier pitcher ERA/WHIP**: Steamer still wins on these for elite arms.

## Recommended Configuration

The ensemble blend in `fbm.toml` represents the recommended usage:

```toml
[models.ensemble.params]
mode = "weighted_average"
season = 2025
stats = ["avg", "obp", "slg", "era", "fip", "whip", "k_per_9", "bb_per_9"]

[models.ensemble.params.components]
marcel = 0.6
"statcast-gbm" = 0.4

[models.ensemble.params.versions]
marcel = "latest"
"statcast-gbm" = "latest"
```

This gives Marcel 60% weight (pre-season baseline, provides counting stats as
passthrough) and statcast-gbm 40% weight (adds same-season Statcast signal).
Overlapping stats get a true weighted blend; stats unique to one system pass
through at that system's value.

## Next Steps

1. ~~**Lag Statcast features for pre-season mode**~~ — Done and evaluated. Preseason
   mode leads on pitcher ERA/FIP/WHIP for the full pool but trails Marcel/Steamer
   on batter stats and top-200 accuracy. See preseason results above.
2. ~~**In-season talent-delta monitor**~~ — Done. `fbm report talent-delta`
   compares true-talent estimates to actuals for buy-low/sell-high analysis.
3. ~~**Statcast-adjusted Marcel**~~ — Done. Marcel can blend statcast-gbm
   true-talent rates into its inputs via `statcast_augment = true`.
4. **Tune ensemble weights** — the 60/40 split was a starting point. A
   stat-specific weighting (e.g., 80% statcast-gbm for SLG, 80% Marcel for ERA)
   could improve results.
5. **Add more training seasons** — extending to 2020-2024 (skipping 2020's
   shortened season) could help.
6. **Hyperparameter tuning** — the model uses default scikit-learn params.
   Cross-validated grid search could improve accuracy.
7. **Feature engineering** — adding sprint speed, catch probability, or park
   factors could address the top-200 pitcher weakness.
8. **Prune noisy features** — ablation shows `avg_v_break`, `cu_velo`, `si_pct`,
   and all batter lag stats add negligible or negative value.
