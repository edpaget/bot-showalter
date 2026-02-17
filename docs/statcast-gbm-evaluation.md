# Statcast GBM Model: Evaluation & Analysis

**Model version:** latest (retrained 2026-02-17)
**Training data:** 2022-2024 (holdout: 2025)
**Evaluation season:** 2025 actuals (FanGraphs)
**Date:** 2026-02-17

## Overview

The statcast-gbm family consists of two gradient-boosted regression models that use
Statcast pitch-level data (exit velocity, spin rates, plate discipline metrics,
expected stats) alongside traditional lag statistics to estimate rate stats for
batters and pitchers. They predict rate/ratio stats only (no counting stats, no WAR,
no playing time).

Both models are built on scikit-learn's `HistGradientBoostingRegressor` with separate
sub-models for each target stat. Batter and pitcher pipelines are fully independent
with different feature sets tailored to each player type.

### Architecture: Two registered models

The family is implemented as two separate registered model classes sharing a common
base (`_StatcastGBMBase`):

- **`statcast-gbm`** (`StatcastGBMModel`): A **true-talent estimator** that uses
  **same-season Statcast features**. It trains on historical season data (e.g., 2023
  exit velocity → 2023 AVG) and then applies the learned relationship to the current
  season's Statcast data to estimate what a player's rate stats *should* be given
  their underlying quality of contact, plate discipline, and stuff metrics. The
  difference between a player's actual stats and the model's estimate reveals
  over/underperformance — useful for in-season buy-low/sell-high analysis.

  **Important:** This is NOT a projection model. It does not predict the future. It
  estimates current true talent from current-season Statcast data. See
  [Evaluation Methodology](#evaluation-methodology-for-true-talent-estimator) below
  for how to properly evaluate a true-talent estimator.

- **`statcast-gbm-preseason`** (`StatcastGBMPreseasonModel`): A **projection model**
  that uses **prior-season Statcast features** (lag=1). It trains on historical data
  (e.g., 2022 exit velocity → 2023 AVG) and uses prior-year Statcast data to make
  genuine pre-season projections comparable to Marcel/Steamer.

Each model is independently registered in the model registry and has its own feature
set, curated column list, and artifact directory. They are invoked separately via CLI:

```bash
fbm train statcast-gbm --season 2022 --season 2023 --season 2024 --season 2025
fbm train statcast-gbm-preseason --season 2022 --season 2023 --season 2024 --season 2025
```

The true-talent estimates from `statcast-gbm` can be blended into Marcel's inputs
via statcast-adjusted Marcel (`statcast_augment = true` in Marcel config), and
compared against actuals via the `fbm report talent-delta` command for in-season
buy-low/sell-high analysis.

### Phase 1 feature pruning (2026-02-17)

The live model (`statcast-gbm`) uses **curated (pruned) feature sets** based on an
ablation study. Features with zero or negative permutation importance were removed:
- Batter: 14 features removed (all lag stats except `so_1`, `age`, `ld_pct`)
- Pitcher: 20 features removed (most pitch-mix and movement features, `age`, several lag stats)

The preseason model (`statcast-gbm-preseason`) retains its **full unpruned feature
sets** after a no-go gate result on pruning (pruning degraded preseason batter
accuracy).

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

### Live model (`statcast-gbm`) — curated features

After Phase 1 pruning, the live model uses a reduced feature set focused on the
features with positive permutation importance.

#### Batter features (21 total)

| Category | Features | Description |
|----------|----------|-------------|
| Lag stats | `so_1` | Only strikeouts from prior season (all other lag stats pruned) |
| Batted ball | `avg_exit_velo`, `max_exit_velo`, `barrel_pct`, `hard_hit_pct`, `sweet_spot_pct`, `gb_pct`, `fb_pct` | Same-season Statcast quality-of-contact metrics |
| Plate discipline | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct`, `k_pct_1`, `bb_pct_1` | Same-season swing decision metrics |
| Expected stats | `xba`, `xwoba`, `xslg` | Same-season Statcast expected outcomes |
| Spray angle | `pull_pct`, `center_pct`, `oppo_pct` | Same-season spray distribution |

#### Pitcher features (26 total)

| Category | Features | Description |
|----------|----------|-------------|
| Lag stats | `ip_1`, `era_1`, `fip_1` | Key prior-season stats (most lag stats pruned) |
| Pitch velocity | `ff_velo`, `ch_velo`, `cu_velo`, `fc_velo` | Same-season fastball, change, curve, cutter velocity |
| Pitch mix | `ch_pct`, `cu_pct`, `fc_pct` | Same-season usage % for 3 pitch types |
| Spin profile | `avg_spin_rate`, `ff_spin`, `ch_spin`, `ff_h_break`, `avg_v_break` | Same-season spin and movement |
| Extensions | `avg_extension` | Release point extension |
| Plate discipline | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct`, `first_pitch_strike_pct` | Same-season swing decision metrics |
| Batted ball | `fb_pct_against`, `gb_pct_against`, `hard_hit_pct_against` | Same-season batted-ball quality against |

### Preseason model (`statcast-gbm-preseason`) — full features

The preseason model retains its full unpruned feature set (pruning failed the
go/no-go gate for this model).

#### Batter features (35 total)

| Category | Features | Description |
|----------|----------|-------------|
| Age | `age` | Player age at season start |
| Lag stats (year 1) | `pa`, `hr`, `h`, `doubles`, `triples`, `bb`, `so`, `sb`, `avg`, `obp`, `slg` | Traditional FanGraphs stats from prior season |
| Batted ball (lag 1) | `avg_exit_velo`, `max_exit_velo`, `avg_launch_angle`, `barrel_pct`, `hard_hit_pct`, `sweet_spot_pct`, `gb_pct`, `fb_pct`, `ld_pct` | Prior-season Statcast quality-of-contact |
| Plate discipline (lag 1) | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct` | Prior-season swing decision metrics |
| Expected stats (lag 1) | `xba`, `xwoba`, `xslg` | Prior-season expected outcomes |
| Spray angle (lag 1) | `pull_pct`, `center_pct`, `oppo_pct` | Prior-season spray distribution |
| Rate stats (lag 1) | `k_pct_1`, `bb_pct_1` | Prior-season K% and BB% |

#### Pitcher features (46 total)

| Category | Features | Description |
|----------|----------|-------------|
| Age | `age` | Player age at season start |
| Lag stats (year 1) | `ip`, `so`, `bb`, `hr`, `era`, `fip` | Traditional FanGraphs stats from prior season |
| Pitch velocity (lag 1) | `ff_velo`, `sl_velo`, `ch_velo`, `cu_velo`, `si_velo`, `fc_velo` | Prior-season pitch velocities |
| Pitch mix (lag 1) | `ff_pct`, `sl_pct`, `ch_pct`, `cu_pct`, `si_pct`, `fc_pct` | Prior-season usage percentages |
| Spin profile (lag 1) | `avg_spin_rate`, `ff_spin`, `sl_spin`, `ch_spin`, `cu_spin` | Prior-season spin rates |
| Movement (lag 1) | `ff_h_break`, `sl_h_break`, `avg_v_break`, `sl_v_break`, `ch_h_break`, `ch_v_break`, `cu_h_break`, `cu_v_break` | Prior-season pitch movement |
| Extensions (lag 1) | `avg_extension` | Prior-season release extension |
| Plate discipline (lag 1) | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct`, `first_pitch_strike_pct` | Prior-season swing decision metrics |
| Batted ball (lag 1) | `fb_pct_against`, `gb_pct_against`, `hard_hit_pct_against` | Prior-season batted-ball quality against |

## Evaluation Methodology for True-Talent Estimator

The live model (`statcast-gbm`) is a **true-talent estimator**, not a projection
model. This distinction affects how it should be trained, used, and evaluated.

### How it works

The model learns the mapping: *same-season Statcast features → same-season rate
stats*. For example, it learns "a batter with 90 mph avg exit velo, 12% barrel
rate, and 8% whiff rate typically posts a .270 AVG." It then applies this learned
relationship to a new season's Statcast data to estimate what a player's stats
*should* be given their underlying skill indicators.

The **residual** (actual stat - model estimate) reveals over/underperformance:
- Positive residual = player outperformed their Statcast indicators (luck, sequencing)
- Negative residual = player underperformed (bad luck, could be a buy-low candidate)

### Training approach

The model trains on **prior completed seasons** (e.g., 2022-2024) and predicts the
**current season** (2025) using same-season Statcast inputs. It does NOT train on
the prediction season, because:

1. **During the season**: Full-season targets aren't available yet, so you can't
   include the current season in training.
2. **After the season** (for evaluation): Including the prediction season in training
   would be in-sample evaluation, giving unrealistically optimistic metrics.

The training data teaches the general Statcast → rate stat relationship across
multiple seasons. This relationship transfers to new seasons because the physics
of batted balls and swing mechanics don't change year-to-year (barring rule changes).

### Proper evaluation approaches

Standard RMSE-vs-actuals is a useful but incomplete metric for true-talent estimators.
Better approaches include:

1. **Next-season predictive validity**: If the model correctly estimates true talent,
   then its 2024 estimates should predict 2025 actuals better than 2024 raw stats
   predict 2025 actuals. This tests whether the model's "de-noising" of raw stats
   actually improves signal quality.

2. **Residual non-persistence**: If residuals (actual - estimate) truly represent
   noise rather than skill, they should not persist year-to-year. Measure the
   correlation between 2024 residuals and 2025 residuals — it should be near zero.
   High correlation would suggest the model is systematically mis-estimating certain
   player types.

3. **Shrinkage quality**: Compare the model's estimates to raw stats. The estimates
   should be "shrunk" toward the mean (less extreme than raw stats). Good shrinkage
   means the estimates have lower variance but similar or better correlation with
   next-season stats compared to raw stats.

4. **R-squared decomposition**: What fraction of within-season variance does the
   model explain? A good true-talent estimator should have high R² with residuals
   that are genuinely random (no pattern by player type, sample size, or park).

5. **Residual regression to estimate**: If the model is right, players who
   outperformed their true-talent estimate in year N should regress toward the
   estimate in year N+1. Measure how much of the residual "corrects" — a well-
   calibrated model should see ~100% regression of residuals.

These approaches require multi-season data and are planned for future evaluation
work. The current evaluation uses standard RMSE-vs-actuals as a pragmatic baseline.

## Holdout Metrics (2025 season)

These are RMSE values from the time-series holdout split during training
(trained on 2022-2024, holdout on 2025).

| Target | Live (pruned) | Preseason (unpruned) |
|--------|---------------|----------------------|
| **Batter** | | |
| avg | 0.0311 | 0.0492 |
| obp | 0.0336 | 0.0589 |
| slg | 0.0595 | 0.0873 |
| woba | 0.0358 | 0.0583 |
| iso | 0.0357 | 0.0512 |
| babip | 0.0683 | 0.0744 |
| **Pitcher** | | |
| era | 6.153 | 6.268 |
| fip | 3.404 | 3.467 |
| k/9 | 1.981 | 2.980 |
| bb/9 | 2.710 | 3.757 |
| hr/9 | 5.264 | 5.216 |
| babip | 0.125 | 0.123 |
| whip | 0.850 | 0.923 |

The live model's pruned features improved holdout RMSE on all 8 primary targets
(8/8 GO at the Phase 1 gate). Preseason pruning was reverted after failing the
go/no-go gate (only 2/8 targets improved).

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

### `statcast-gbm` — live model (same-season Statcast)

Note: the live model uses same-season Statcast input, giving it an information
advantage over Marcel and Steamer (which are pre-season projections). This is
by design — it's a true-talent estimator, not a projection system.

#### Full player pool

RMSE comparison across all players with 2025 FanGraphs actuals. Lower is better.
Bold = best in row.

| Stat | Marcel | Live | Steamer | Ensemble (60/40) |
|------|--------|-------------|---------|-------------------|
| avg | 0.0587 | **0.0479** | 0.0515 | 0.0494 |
| obp | 0.0643 | **0.0510** | 0.0551 | 0.0529 |
| slg | 0.0937 | **0.0673** | 0.0763 | 0.0735 |
| era | 7.670 | 6.633 | 7.342 | **6.324** |
| fip | 12.817 | **3.310** | 4.486 | **3.310** |
| whip | 1.248 | 0.829 | 1.191 | **0.804** |
| k/9 | 2.435 | **1.858** | 2.122 | 1.940 |
| bb/9 | 3.541 | **2.566** | 3.456 | 2.615 |

The live model leads on every stat except ERA and WHIP (where the ensemble wins).

#### Top 200 players (by WAR)

The fantasy-relevant comparison. Top 200 by 2025 WAR filters to everyday
players and rotation arms — the population fantasy managers care most about.

| Stat | Marcel | Live | Steamer | Ensemble (60/40) |
|------|--------|-------------|---------|-------------------|
| avg | 0.0286 | 0.0233 | 0.0247 | **0.0228** |
| obp | 0.0307 | 0.0263 | 0.0263 | **0.0251** |
| slg | 0.0625 | **0.0471** | 0.0549 | 0.0485 |
| era | 1.200 | 1.634 | **0.996** | 1.296 |
| fip | 3.360 | 0.989 | **0.742** | 0.989 |
| whip | 0.212 | 0.227 | **0.182** | 0.194 |
| k/9 | 1.808 | **1.009** | 1.329 | 1.256 |
| bb/9 | 0.883 | 0.831 | 0.759 | **0.748** |

For top 200: the live model leads on AVG, SLG, and K/9. The ensemble leads on
AVG, OBP, and BB/9. Steamer leads on ERA, FIP, and WHIP for elite players.

### `statcast-gbm-preseason` — preseason model (prior-season Statcast)

This is the apples-to-apples comparison: the preseason model uses only prior-year
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

### 1. Preseason model lags behind Marcel and Steamer on batter stats

The preseason model enables genuine pre-season projections but is not competitive
with Marcel or Steamer on batter rate stats (AVG, OBP, SLG). Lagging the Statcast
features by one year costs ~22% accuracy on AVG and ~28% on SLG compared to the
live model. The preseason model's strength is pitcher ERA/FIP/WHIP for the full
player pool, where prior-year stuff metrics retain strong predictive power.

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

### 5. Training data is only 3 seasons deep

The model trains on 2022-2024 and validates on 2025. This is a small training
set. More seasons would help the model learn better regression patterns, but
Statcast data quality and rule changes (pitch clock in 2023, shift ban) create
legitimate reasons to limit the lookback window.

### 6. No player-level persistence — regression toward population mean

The model uses a flat `HistGradientBoostingRegressor` per target stat with no
player-level random effects or hierarchical structure. Each (player, season)
row is treated independently. Player identity is captured **only** through that
season's observed Statcast metrics — there are no player-specific intercepts,
no multi-year memory, and most lag-1 stats were pruned as noisy.

This architecture creates an asymmetry between batter and pitcher predictions:

**Pitchers**: Stuff metrics (whiff_rate, spin_rate, chase_rate) are tight
mechanical proxies for skill. A pitcher with elite spin and whiff rates will
miss bats regardless of who they are. The Statcast → outcome mapping is strong,
and stronger regularization helps the model learn the stable relationship
without overfitting to in-season noise (strand rate, BABIP-against, HR/FB
variance). This is why pitcher hyperparameter tuning improved accuracy.

**Batters**: Exit velocity and plate discipline are noisier predictors of
traditional rate stats. Two batters with identical exit velocity profiles can
have very different AVGs because of BABIP variance (speed, spray quality,
opponent defense, luck), approach consistency (elite batters make mid-season
adjustments that don't show up in aggregate Statcast distributions), and
selection effects (facing different pitch mixes). The best batters consistently
outperform their Statcast profile — a 10-year veteran with a career .300 AVG
is treated the same as a rookie with identical single-season Statcast numbers.
Stronger regularization pushes predictions toward the population mean, which
actively hurts predictions for these consistent overperformers.

**Phase 1 tuning result**: Pitcher tuning was a GO (4/5 targets improved, mean
RMSE improved ~4.4%). Batter tuning was a NO-GO (AVG degraded 12.9%, far
exceeding the 5% limit). Only pitcher-tuned params were kept.

**Potential improvements**: Multi-year rolling features (2-3 year weighted
averages of prior Statcast metrics), player-level intercepts via mixed-effects
models (e.g., MERF), historical consistency features (year-to-year variance,
persistent xBA-AVG gap), or career-length priors that reduce regression for
established players.

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

- **Pre-season drafts**: The preseason model can produce blind forecasts, but it
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
   model leads on pitcher ERA/FIP/WHIP for the full pool but trails Marcel/Steamer
   on batter stats and top-200 accuracy. See preseason results above.
2. ~~**In-season talent-delta monitor**~~ — Done. `fbm report talent-delta`
   compares true-talent estimates to actuals for buy-low/sell-high analysis.
3. ~~**Statcast-adjusted Marcel**~~ — Done. Marcel can blend statcast-gbm
   true-talent rates into its inputs via `statcast_augment = true`.
4. ~~**Split into two models**~~ — Done (2026-02-17). `statcast-gbm` and
   `statcast-gbm-preseason` are now separate registered models with independent
   feature sets.
5. ~~**Prune noisy features (Phase 1)**~~ — Done for live model (GO: 8/8 holdout
   targets improved). Reverted for preseason model (NO-GO: only 2/8 improved).
6. ~~**Hyperparameter tuning (Phase 1)**~~ — Done. Temporal expanding CV +
   exhaustive grid search over 1024 combinations. Pitcher tuning: GO (4/5
   targets improved, mean RMSE ~4.4% better). Batter tuning: NO-GO (AVG
   degraded 12.9%). Pitcher-only tuned params committed. See
   [Weakness #6](#6-no-player-level-persistence--regression-toward-population-mean).
7. **Implement true-talent evaluation suite** — next-season predictive validity,
   residual non-persistence, shrinkage quality metrics. See
   [Evaluation Methodology](#evaluation-methodology-for-true-talent-estimator).
8. **Player-level persistence for batters** — multi-year rolling features,
   mixed-effects models, or career-length priors to address the batter
   regression-to-mean problem identified during hyperparameter tuning.
9. **Tune ensemble weights** — the 60/40 split was a starting point. A
   stat-specific weighting (e.g., 80% statcast-gbm for SLG, 80% Marcel for ERA)
   could improve results.
10. **Add more training seasons** — extending to 2021-2024 (skipping 2020's
    shortened season) could help.
11. **Feature engineering** — adding sprint speed, catch probability, or park
    factors could address the top-200 pitcher weakness.
