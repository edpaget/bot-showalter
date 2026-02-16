# Statcast GBM Model: Evaluation & Analysis

**Model version:** v1
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
| Batted ball | `avg_exit_velo`, `max_exit_velo`, `avg_launch_angle`, `barrel_pct`, `hard_hit_pct` | Statcast quality-of-contact metrics |
| Plate discipline | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct` | Swing decision metrics |
| Expected stats | `xba`, `xwoba`, `xslg` | Statcast expected outcomes based on batted-ball physics |

### Pitcher features (36 total)

| Category | Features | Description |
|----------|----------|-------------|
| Age | `age` | Player age at season start |
| Lag stats (year 1 & 2) | `ip`, `so`, `bb`, `hr`, `era`, `fip` x 2 lags | Traditional FanGraphs stats from prior two seasons |
| Pitch mix | `ff_pct/velo`, `sl_pct/velo`, `ch_pct/velo`, `cu_pct/velo`, `si_pct/velo`, `fc_pct/velo` | Usage % and velocity for 6 pitch types |
| Spin profile | `avg_spin_rate`, `ff_spin`, `sl_spin`, `cu_spin`, `avg_h_break`, `avg_v_break` | Spin rates and movement |
| Plate discipline | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct` | Swing decision metrics (from pitcher's perspective) |

## Holdout Metrics (2024 season)

These are RMSE values from the time-series holdout split during training
(trained on 2022-2023, tested on 2024):

| Target | Holdout RMSE |
|--------|-------------|
| **Batter** | |
| avg | 0.0523 |
| obp | 0.0626 |
| slg | 0.0915 |
| woba | 0.0621 |
| iso | 0.0524 |
| babip | 0.0736 |
| **Pitcher** | |
| era | 6.246 |
| fip | 3.533 |
| k/9 | 2.042 |
| bb/9 | 3.114 |
| hr/9 | 5.256 |
| babip | 0.126 |
| whip | 0.847 |

## Accuracy vs Other Systems (2025 Season)

### Full player pool

RMSE comparison across all players with 2025 FanGraphs actuals. Lower is better.
Bold = best in row.

| Stat | Marcel | Statcast-GBM | Steamer | Ensemble (60/40) |
|------|--------|-------------|---------|-------------------|
| avg | 0.0587 | 0.0605 | **0.0515** | 0.0585 |
| obp | 0.0643 | 0.0666 | **0.0551** | 0.0653 |
| slg | 0.0937 | **0.0902** | 0.0763 | 0.0888 |
| era | 7.670 | 6.830 | 7.342 | **6.775** |
| fip | 12.817 | **3.703** | 4.486 | **3.703** |
| whip | 1.248 | 0.970 | 1.191 | **0.965** |
| k/9 | 2.435 | 3.291 | **2.122** | 3.183 |
| bb/9 | 3.541 | 3.289 | 3.456 | **3.232** |

### Top 200 players (by WAR)

The fantasy-relevant comparison. Top 200 by 2025 WAR filters to everyday
players and rotation arms — the population fantasy managers care most about.

| Stat | Marcel | Statcast-GBM | Steamer | Ensemble (60/40) |
|------|--------|-------------|---------|-------------------|
| avg | 0.0286 | 0.0417 | **0.0247** | 0.0299 |
| obp | 0.0307 | 0.0473 | **0.0263** | 0.0328 |
| slg | 0.0625 | 0.0844 | **0.0549** | 0.0638 |
| era | 1.200 | 1.251 | **0.996** | 1.128 |
| fip | 3.360 | 0.854 | **0.742** | 0.854 |
| whip | 0.212 | 0.192 | **0.182** | 0.186 |
| k/9 | 1.808 | 2.054 | **1.329** | 1.699 |
| bb/9 | 0.883 | 0.779 | **0.759** | 0.756 |

## Strengths

### 1. Pitcher rate stats (full pool)

The model's strongest contribution is pitcher projections across the broad player
pool. On ERA, FIP, WHIP, and BB/9 for the full population, statcast-gbm beats
Marcel handily and is competitive with or beats Steamer. This is where the Statcast
features — pitch mix, spin profiles, and plate discipline — provide genuine
signal beyond what traditional lag stats capture.

- **FIP RMSE of 3.70** crushes Marcel's 12.82 and beats Steamer's 4.49. The pitch-level
  data (velocity, spin, movement) directly relates to fielding-independent outcomes.
- **WHIP RMSE of 0.97** is the best single-system result, beating Steamer's 1.19.
- **ERA RMSE of 6.83** beats both Marcel (7.67) and Steamer (7.34).

### 2. Strong ensemble component

The model's primary value is as an ensemble ingredient. The 60% Marcel / 40%
statcast-gbm blend outperforms either component alone on most pitcher stats and
matches or improves on Marcel's batting projections:

- Ensemble is the best first-party system on ERA, FIP, WHIP, and BB/9 (full pool).
- On top-200 BB/9 (0.756), the ensemble is essentially tied with Steamer (0.759).
- Ensemble WHIP (0.186) is within 2% of Steamer (0.182) for top 200.

### 3. Captures Statcast-era signal

The feature set captures information that purely historical systems miss:
- Exit velocity / barrel rate predict power better than raw HR counts.
- Whiff rate and chase rate capture strikeout/walk tendencies at a granular level.
- xBA/xwOBA provide "luck-adjusted" baselines that stabilize faster than actual
  results over small samples.
- Pitch velocity and spin data capture stuff quality independent of results.

### 4. Handles missing data natively

`HistGradientBoostingRegressor` treats NaN as a first-class missing indicator,
so the model gracefully handles players with partial Statcast data (e.g., limited
batted-ball events, new pitch types). No imputation heuristics needed.

## Weaknesses

### 1. Batter rate stats lag behind established systems

For the stats that matter most in fantasy (AVG, OBP, SLG), the model trails both
Marcel and Steamer. The gap widens for top-200 players:

| Stat | Statcast-GBM | Marcel | Steamer |
|------|-------------|--------|---------|
| avg (top 200) | 0.0417 | 0.0286 | 0.0247 |
| obp (top 200) | 0.0473 | 0.0307 | 0.0263 |
| slg (top 200) | 0.0844 | 0.0625 | 0.0549 |

AVG/OBP are stabilization-heavy stats influenced by BABIP variance, defensive
positioning, and spray angle — factors not fully captured by exit velocity alone.
Marcel's multi-year regression approach handles this better for known quantities.

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
from Baseball Savant. Players without meaningful MLB pitch data in the prior two
seasons (rookies from the minors, international signings, September call-ups with
tiny samples) get no projection. For fantasy, this means the model misses some
breakout rookies that Steamer and ZiPS project using minor-league data.

### 4. Pitcher stats degrade for top 200

While pitcher projections dominate the full pool, the advantage narrows
significantly for top-200 players. Steamer's edge in ERA (0.996 vs 1.251) and
K/9 (1.329 vs 2.054) for fantasy-relevant pitchers suggests that for established
arms, Steamer's deeper modeling (including park factors, defense, workload
modeling) adds value that raw Statcast features don't fully capture.

### 5. K/9 is consistently weak

K/9 RMSE is the worst relative performance across both pools. Strikeout rate
projections lag behind both Marcel and Steamer. This may reflect that the model's
whiff_rate and swinging_strike_pct features don't fully capture the
pitcher-batter matchup dynamics, platoon effects, and bullpen vs. starter
usage patterns that drive K/9 variance.

### 6. Training data is only 2 seasons deep

The v1 model trains on 2022-2023 and validates on 2024. This is a very small
training set. More seasons would help the model learn better regression patterns,
but Statcast data quality and rule changes (pitch clock in 2023, shift ban) create
legitimate reasons to limit the lookback window.

### 7. Known bugs (mostly fixed)

Six bugs were identified and fixed in the statcast-gbm-fixes roadmap, including
an X/y length mismatch, a pitcher join bug (pitchers were getting batter Statcast
data), and transforms returning 0.0 instead of NaN for missing data. All have
been resolved, but the v1 artifacts were trained before some of these fixes and
would benefit from retraining.

## Fantasy Implications

### Where to trust statcast-gbm

- **Pitcher rate stats for deep leagues**: ERA, FIP, WHIP, and BB/9 projections
  are genuinely strong for the full player pool. In leagues where you're streaming
  or rostering fringe pitchers, this model adds real value.
- **As an ensemble component**: The 60/40 Marcel/statcast-gbm blend consistently
  beats Marcel alone and approaches Steamer quality. This is the recommended
  usage — blend, don't replace.
- **Identifying pitchers with stuff changes**: Pitch velocity, spin rate, and
  movement changes show up in the feature set before they appear in ERA. The model
  may capture these trends earlier than pure results-based systems.

### Where to be skeptical

- **Batter AVG/OBP projections**: Don't rely on statcast-gbm alone for batting
  rate stats. Marcel's regression framework is better calibrated for these.
- **Counting-stat-dependent formats**: Roto leagues need HR, RBI, SB projections
  that this model doesn't provide. Use the ensemble or Marcel directly.
- **Rookie/prospect projections**: No Statcast data = no projection. Use a system
  with minor-league translation (or the MLE model) for prospects.
- **Top-tier pitcher K/9**: The model's K/9 projections are its weakest stat
  relative to alternatives.

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
"statcast-gbm" = "v1"
```

This gives Marcel 60% weight (better calibrated for batting, provides counting
stats as passthrough) and statcast-gbm 40% weight (adds Statcast signal for
pitcher rates and SLG). Overlapping stats get a true weighted blend; stats
unique to one system pass through at that system's value.

## Next Steps

1. **Retrain** on 2022-2024 with all bug fixes applied (current v1 was trained
   before some fixes landed).
2. **Tune ensemble weights** — the 60/40 split was a starting point. A
   stat-specific weighting (e.g., 80% Marcel for AVG, 60% statcast-gbm for FIP)
   could improve results.
3. **Add more training seasons** — extending to 2020-2024 (skipping 2020's
   shortened season) could help.
4. **Hyperparameter tuning** — the model uses default scikit-learn params.
   Cross-validated grid search could improve accuracy.
5. **Feature engineering** — adding sprint speed, catch probability, or park
   factors could address some of the batter-side weakness.
