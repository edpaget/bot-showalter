# Batter Persistence Roadmap

**Model:** `statcast-gbm` (primary), `statcast-gbm-preseason` (secondary beneficiary)
**Created:** 2026-02-17
**Updated:** 2026-02-17 (Phase 1 results)
**Status:** Phase 1 complete — NO-GO on persistence-driven approach; pivoting to
standalone feature engineering.
**Goal:** ~~Address the batter regression-to-mean problem identified during Phase 2
hyperparameter tuning, where the model lacks player-level persistence and treats
all batters with identical Statcast profiles identically regardless of track record.~~
Improve batter predictions through targeted feature engineering (Phases 2-4),
evaluated on their own merits rather than as fixes for a persistence gap.

## Problem Statement

The statcast-gbm model uses a flat `HistGradientBoostingRegressor` per target with
no player-level structure. Each (player, season) row is treated independently. Player
identity is captured only through that season's observed Statcast metrics — there are
no player-specific intercepts, no multi-year memory, and most lag-1 stats were pruned.

This architecture causes predictions to regress toward the population mean, which
actively hurts accuracy for batters who consistently over- or underperform their
Statcast profile. A 10-year veteran with a career .300 AVG is treated the same as
a rookie with identical single-season Statcast numbers.

### Why batters are affected more than pitchers

Pitcher outcomes (K/9, BB/9, FIP) are tightly coupled to mechanical Statcast
metrics (whiff rate, spin rate, chase rate). The Statcast-to-outcome mapping is
strong enough that player identity adds little signal — hence pitcher hyperparameter
tuning succeeded (GO) while batter tuning failed (NO-GO).

Batter outcomes depend on skills that Statcast expected stats deliberately exclude:

- **Spray direction within batted ball types** — Pulled fly balls produce wOBA of
  .866 vs xwOBA of .672 (+194 pts); center fly balls produce .312 vs .520 (-208
  pts). Player-level adjustments range from +33 to -88 points per season.
- **Sprint speed on ground balls** — Adding sprint speed to xBA improved predictive
  R-squared from .17 to .23. The effect is concentrated on ground balls.
- **Park effects** — Coors Field (1.120 park factor) alone explains multi-year
  overperformance of +20 pts for Blackmon, Arenado, Story across 5+ consecutive
  seasons.

### Evidence of persistent skill (pre-diagnostic research)

The wOBA-xwOBA gap has a year-to-year correlation of r = 0.43 (R-squared = 0.17).
Roughly 17% of a player's overperformance in year N carries into year N+1. This is
a non-trivial skill signal that raw xStats discard.

BABIP — the noisiest batter rate stat — has a year-to-year correlation of r ~ 0.37
for hitters. Predictors of persistent high/low BABIP: line drive rate, hard-hit rate,
opposite-field ground ball rate, sprint speed, and infield fly ball rate.

**Phase 1 update:** These research numbers describe persistence in *raw stat gaps*
(wOBA vs xwOBA, raw BABIP year-to-year). The model already incorporates many of
the underlying Statcast features as inputs, so it absorbs much of this signal. The
diagnostic (below) confirms that *model residuals* are near-random — the persistence
the research identified is already being captured, not discarded.

### Relationship to existing roadmaps

The live model roadmap Phase 4 (New Batter Features) covers sprint speed, park
factors, batted-ball interactions, and platoon splits as individual feature additions.
This roadmap addresses the **structural problem** — giving the model player-level
memory and persistence — which is complementary. Some feature work here (spray-by-
batted-ball-type, career experience) is unique to this roadmap; architectural changes
(MERF, multi-year rolling features) are entirely new.

---

## Phase 1: Batter Residual Persistence Diagnostic

**Priority:** High — quantify the problem before attempting fixes.
**Prerequisite:** Trained `statcast-gbm` models for at least 2 consecutive seasons
(2024 and 2025 predictions against actuals).

**Rationale:** Before adding features or changing architecture, we need to measure
the magnitude and structure of the batter regression-to-mean problem. Which stats
are most affected? Which player archetypes are chronic over/underperformers? How
large is the residual persistence signal?

### Work

Build a diagnostic analysis (script or notebook) that:

1. **Compute per-player residuals** for each batter target (avg, obp, slg, woba,
   iso, babip) across consecutive seasons. Residual = actual - model estimate.

2. **Measure year-to-year residual correlation** for returning batters:
   - Pearson r between 2024 residuals and 2025 residuals, per target
   - Compare to baseline: year-to-year correlation of raw stats (actual 2024 vs
     actual 2025) to assess whether the model's residuals are more or less
     persistent than raw stat noise
   - Break down by PA buckets (< 200, 200-400, 400+) to see if persistence
     scales with sample size

3. **Identify chronic over/underperformers**:
   - Rank batters by mean residual magnitude across seasons
   - Characterize the top/bottom quintiles: sprint speed, pull tendency,
     home park factor, career length, handedness
   - Flag players with residuals > 1 standard deviation in the same direction
     for 2+ consecutive seasons

4. **Quantify the ceiling** for improvement:
   - If we could perfectly predict each player's persistent residual component,
     how much would batter RMSE improve? This is an upper bound on the
     improvement available from player-level persistence features.
   - Compute: RMSE(actuals, model + mean_residual) vs RMSE(actuals, model)
     where mean_residual is the player's average residual from prior seasons

### Deliverable

A summary document with:
- Residual correlation table (per stat, per PA bucket)
- List of chronic over/underperformers with their characteristics
- Estimated RMSE ceiling improvement per stat
- Recommendation on which phases to prioritize based on findings

### Go/No-Go Gate

This phase produces diagnostics, not model changes. The gate determines whether
the problem is worth pursuing:

- **Go:** Year-to-year residual correlation > 0.10 for at least 3/6 batter stats
  AND ceiling RMSE improvement > 2% for at least 2/6 stats.
- **No-go:** If residuals are already near-random (r < 0.10 for all stats), the
  model is correctly estimating true talent and the remaining error is genuinely
  unpredictable noise. Abandon this roadmap and focus on feature engineering
  (live roadmap Phase 4) instead.

### Results (2024 -> 2025, statcast-gbm/latest, N=961 returning batters)

**Verdict: NO-GO.** Persistence passes: 1/6. Ceiling passes: 0/6.

#### Residual correlation (year-over-year)

| Stat  | Overall r | <200 PA | 200-400 | 400+ | N Ret |
|-------|-----------|---------|---------|------|-------|
| avg   | -0.016    | -0.043  | -0.022  | 0.053 | 961  |
| obp   | -0.026    | -0.072  |  0.152  | 0.117 | 961  |
| slg   |  0.033    |  0.008  | -0.031  | 0.169 | 961  |
| woba  | -0.011    | -0.055  |  0.085  | 0.142 | 961  |
| iso   |  0.106    |  0.075  |  0.132  | 0.190 | 961  |
| babip | -0.053    | -0.067  | -0.063  | 0.058 | 463  |

Only ISO (r = 0.106) passes the persistence threshold. Five of six stats have
overall correlations indistinguishable from zero.

Notable pattern: the 400+ PA bucket shows elevated correlations across several
stats (obp: 0.117, slg: 0.169, woba: 0.142, iso: 0.190). This suggests a weak
signal exists for established regulars but is diluted by part-time player noise
in the overall numbers.

#### RMSE improvement ceiling

| Stat  | RMSE Base | RMSE Corr | Improvement |
|-------|-----------|-----------|-------------|
| avg   | 0.0518    | 0.0603    | -16.4%      |
| obp   | 0.0521    | 0.0612    | -17.4%      |
| slg   | 0.0649    | 0.0810    | -24.8%      |
| woba  | 0.0484    | 0.0579    | -19.5%      |
| iso   | 0.0317    | 0.0398    | -25.7%      |
| babip | 0.0670    | 0.0934    | -39.4%      |

Naively applying prior-season residuals makes predictions *worse* for all six
stats. The residuals contain more noise than signal — carrying them forward
adds error rather than removing it.

#### Chronic performers

The diagnostic identifies recognizable names among chronic overperformers
(Aaron Judge, Bryce Harper, Shohei Ohtani, Trea Turner, Christian Yelich) and
underperformers (Jack Suwinski, Nolan Jones, Edouard Julien). However, many
chronic performers have very low PA counts (< 50), indicating small-sample
noise dominates the tails. ISO has the most chronic performers (11 over, 48
under), consistent with it being the only stat with meaningful persistence.

#### Interpretation

The research cited above (wOBA-xwOBA gap r = 0.43, BABIP r = 0.37) describes
persistence in *raw stat gaps* and *raw stats*. The model already incorporates
the Statcast features underlying those gaps (exit velocity, barrel rate, sprint
speed, etc.) as inputs. The near-zero residual correlations confirm the model
is successfully capturing the persistent component of those signals — the
remaining error is genuinely unpredictable noise.

This does not contradict the research findings — it shows the model is already
doing its job. The batter tuning failure (SLG degraded 13.6%) was likely caused
by overfitting under stronger regularization, not by a missing persistence
signal.

#### Impact on subsequent phases

- **Phase 5 (MERF) is deprioritized.** Player-level random intercepts address
  persistent residuals, but residuals are not persistent. MERF would learn noise.
- **Phases 2-4 remain viable** but are reframed as standalone feature engineering
  improvements, not fixes for a persistence gap. The justification shifts from
  "close the residual persistence gap" to "add complementary signal the model's
  current features don't capture."
- **Phase 2 (spray-by-batted-ball-type)** is the strongest remaining candidate:
  ISO is the only stat with persistent residuals, and spray-by-batted-ball-type
  interactions are the primary driver of ISO overperformance. The 400+ PA bucket
  signal also suggests these features may help for established hitters.
- **Phase 3 (multi-year rolling features)** may still help by reducing noise in
  single-season Statcast inputs, even though the persistence framing is weakened.
- **Phase 4 (career experience)** has the weakest remaining justification, since
  the model is not systematically under-regressing established players.

---

## Phase 2: Spray-by-Batted-Ball-Type Features

**Priority:** High — strongest remaining candidate after Phase 1 NO-GO. ISO is the
only stat with persistent residuals, and spray interactions are the primary driver.
**Applies to:** Both `statcast-gbm` and `statcast-gbm-preseason`.

**Rationale:** The model already uses overall spray percentages (`pull_pct`,
`center_pct`, `oppo_pct`), but the spray-outcome relationship depends heavily on
batted ball type. Two batters with identical overall pull rates can have very
different outcomes if one pulls fly balls (home runs) while the other pulls ground
balls (double plays). This interaction is the largest known source of systematic
xStats bias (up to +/- 200 points on specific batted ball types).

**Phase 1 context:** ISO was the only stat to pass the persistence threshold
(r = 0.106), and the 400+ PA bucket showed elevated correlations for power-related
stats (slg: 0.169, iso: 0.190). Spray-by-batted-ball-type features directly
target the mechanism behind ISO persistence — pull-heavy fly ball hitters
systematically outperform expected stats in a way the model's current features
don't capture. This phase is now justified by the ISO signal rather than a broad
persistence gap.

### Work

1. **Add spray-by-batted-ball-type features** to the batter feature set:
   - `pull_fb_pct` — fraction of fly balls pulled (HR power signal)
   - `center_fb_pct` — fraction of fly balls to center (weak power signal)
   - `oppo_fb_pct` — fraction of fly balls to opposite field
   - `pull_gb_pct` — fraction of ground balls pulled (double play risk)
   - `oppo_gb_pct` — fraction of ground balls to opposite field (speed/BABIP
     signal)

2. **Source the data:** These require combining Statcast batted ball type
   (ground_ball, fly_ball, line_drive) with spray angle buckets. Check if
   Baseball Savant or the existing Statcast pipeline provides these directly,
   or if they need to be derived from pitch-level data.

3. **Evaluate via ablation:** Run permutation importance on the new features.
   Measure impact on all 6 batter targets.

4. **Apply to preseason model** with lag=1 if the features show year-to-year
   stability. Spray tendencies within batted ball types should be moderately
   persistent (more stable than raw BABIP, less stable than exit velocity).

### Go/No-Go Gate

- **Go:** At least 2 of the 5 new features have permutation importance >= 0.005.
  Batter RMSE improves on >= 3/6 targets without degrading any target > 3%.
- **No-go:** Remove features that don't meet the importance threshold. If none
  pass, the overall spray percentages already capture sufficient directional
  information and the interaction adds noise.

---

## Phase 3: Multi-Year Rolling Features

**Priority:** Medium — gives the model multi-year memory without architectural
changes.
**Applies to:** Primarily `statcast-gbm`, secondarily `statcast-gbm-preseason`.

**Rationale:** The model currently sees only same-season Statcast data (plus `so_1`
after pruning). A 10-year veteran and a first-year player with identical 2025
Statcast profiles get identical predictions. Multi-year rolling features encode
player history directly, letting the model learn that batters with consistently
high exit velocity across 2-3 years are more likely to sustain power production
than a single-season spike.

Research shows the optimal weighting for multi-year averaging is approximately
6/2/1 (most recent season dominates), per Bayesian Marcel analysis. This is far
more recency-biased than Marcel's traditional 5/4/3 weighting.

**Phase 1 context:** The persistence framing is weakened — residuals are already
near-random, so multi-year memory won't help the model "remember" player-specific
biases. However, rolling features may still help by *denoising* single-season
Statcast inputs. A 2-year weighted average of exit velocity is a more reliable
estimate of true contact quality than a single season, which could reduce
prediction variance even without a persistence gap to close.

### Work

1. **Add rolling weighted averages** of key Statcast features:
   - 2-year weighted average (70/30 current/prior) for: `avg_exit_velo`,
     `barrel_pct`, `hard_hit_pct`, `whiff_rate`, `chase_rate`, `xba`, `xwoba`
   - Name convention: `avg_exit_velo_2yr`, `barrel_pct_2yr`, etc.
   - For the live model, this means combining same-season + lag-1 values
   - For the preseason model, this means combining lag-1 + lag-2 values

2. **Add year-to-year delta features** for the most important Statcast metrics:
   - `avg_exit_velo_delta` = current - prior (improving vs declining power)
   - `whiff_rate_delta` = current - prior (improving vs declining contact)
   - `barrel_pct_delta` = current - prior
   - These capture trajectory (breakout vs decline) that static features miss

3. **Add consistency features** measuring year-to-year stability:
   - `xba_avg_gap_prior` = prior season's (xBA - AVG) — persistent over/
     underperformance signal (r = 0.43 year-to-year)
   - `xwoba_woba_gap_prior` = prior season's (xwOBA - wOBA) — same signal
     for overall offense

4. **Implementation:** Extend the `TransformFeature` lag system or add a new
   `RollingTransformFeature` type that the assembler handles by joining multiple
   lagged seasons, applying weights, and averaging. Training data must include
   at least one additional prior season (requires lag-2 data availability).

### Go/No-Go Gate

- **Go:** At least 3 of the new feature groups (rolling averages, deltas,
  consistency) have permutation importance >= 0.003 for at least one batter
  target. Batter RMSE improves on >= 3/6 targets. No target degrades > 5%.
- **No-go:** The model already captures sufficient temporal signal from
  same-season features. Remove groups that don't meet importance thresholds.
  Consider whether the consistency features alone (xBA-AVG gap) add value
  even if rolling averages don't.

---

## Phase 4: Career Experience Features

**Priority:** Low (downgraded from Medium) — weakest remaining justification after
Phase 1 results.
**Applies to:** Both models.

**Rationale:** The model has no concept of player experience. A career .300 hitter
in year 10 and a rookie with the same Statcast profile get the same prediction.
Career experience features provide a prior: established players with long track
records should be regressed less toward the population mean because their observed
performance is a more reliable signal of true talent.

Marcel handles this with a fixed regression formula (denominator of 1200 PA).
ZiPS uses 4-year weighted history (8/5/4/3). This phase adds experience signal
to the GBM without imposing a specific regression formula — the model learns
the relationship from data.

**Phase 1 context:** The diagnostic showed the model is not systematically
under-regressing established players — residuals are near-random across all PA
buckets. The chronic overperformer list includes both veterans (Judge, Altuve,
Harper) and low-PA players in roughly equal proportion, suggesting experience
alone does not predict residuals. Career features may still reduce noise for
the preseason model (where all inputs are lagged), but the live model case is
weaker than originally assumed.

### Work

1. **Career length features:**
   - `mlb_seasons` — number of MLB seasons with >= 100 PA
   - `career_pa` — total career plate appearances entering the season

2. **Career rate stat features** (entering the current season):
   - `career_avg`, `career_obp`, `career_slg` — lifetime rates
   - These serve as player-specific priors that differ from the current season's
     Statcast-derived estimate

3. **Career Statcast baselines** (if multi-year Statcast history available):
   - `career_xba` — lifetime xBA (how good is this batter's contact quality
     historically?)
   - `career_xba_avg_gap` — lifetime (xBA - AVG) gap (does this batter
     chronically over/underperform expected stats?)

4. **Source the data:** Career stats require aggregating across all available
   seasons in the database. May need a new data source or SQL aggregation in
   the assembler. Career Statcast baselines require Statcast data going back
   to at least 2019-2020.

### Go/No-Go Gate

- **Go:** At least 2 career features have permutation importance >= 0.005 for
  at least one batter target. Batter RMSE improves on >= 3/6 targets.
  No target degrades > 3%.
- **No-go:** Career features may be collinear with multi-year rolling features
  from Phase 3. If Phase 3 already captures the signal, career features add
  complexity without value. Remove and move to Phase 5.

---

## Phase 5: Mixed-Effects Random Forest (MERF)

**Priority:** Deprioritized — Phase 1 diagnostic showed residuals are not persistent,
which undercuts the core justification for player-level random intercepts. Revisit
only if Phases 2-4 reveal persistent residual structure that feature engineering
alone cannot capture.
**Applies to:** `statcast-gbm` (primary). Evaluate for preseason if successful.

**Rationale:** Even with better features, the GBM architecture fundamentally
cannot learn player-specific intercepts. Two batters with identical feature vectors
will always get identical predictions. MERF (Mixed Effects Random Forest) wraps the
existing GBM in a mixed-effects framework that adds a player-level random intercept:

```
y = GBM(X) + b_player + epsilon
```

Where `b_player` is a player-specific random effect estimated via EM algorithm.
For known players (seen in training), the learned intercept corrects predictions
by the player's historical residual pattern. For new players, it falls back to
the pure GBM prediction. This directly addresses the core problem: persistent
over/underperformers get player-specific adjustments without hand-crafted features.

### Evidence from baseball research

- Jim Albert's hierarchical beta-binomial model for predicting AVG outperformed
  both raw batting average and xBA, demonstrating the power of partial pooling
  across players.
- FanGraphs Shape+ pitch model used a single `(1 | PitcherID)` random effect
  to capture "unobserved, pitcher-specific variations" not explained by physical
  pitch characteristics. Achieved correlation of 0.868 between predicted and
  actual run values.
- The `merf` Python library (v1.0) supports scikit-learn estimators including
  `HistGradientBoostingRegressor` as the fixed-effect model.

### Work

1. **Evaluate MERF feasibility:**
   - Install `merf` library, verify compatibility with
     `HistGradientBoostingRegressor`
   - Assess training time overhead (MERF uses EM iterations, typically 15-30
     iterations wrapping the base estimator)
   - Determine minimum seasons per player needed for stable random effect
     estimation (likely 2+ seasons with >= 200 PA)

2. **Implement MERF training pipeline:**
   - Wrap the per-target GBM training in MERF's EM framework
   - Player ID becomes the cluster variable (grouping factor)
   - Fixed effects: existing Statcast features (X matrix)
   - Random effects: player-level intercept (Z = 1, simplest case)
   - Store both the GBM model and the player random effects dict in artifacts

3. **Implement MERF prediction pipeline:**
   - For returning players: prediction = GBM(X) + b_player
   - For new players: prediction = GBM(X) (graceful fallback)
   - Requires tracking which players were in the training set

4. **Evaluate against feature-only baseline (post-Phases 2-4):**
   - Compare MERF batter RMSE to the best feature-engineering-only model
   - Check whether MERF's random intercepts capture signal beyond what
     multi-year rolling features and career experience already provide
   - Measure: what fraction of the Phase 1 diagnostic ceiling does MERF close?

5. **Evaluate for preseason model:**
   - If MERF improves the live model, test on preseason model where player
     persistence may matter even more (lagged features are noisier)

### Risks

- **Overfitting with small clusters:** Players with only 1 season of data get
  unreliable random effects. Mitigation: set a minimum cluster size (2+ seasons)
  and fall back to pure GBM for small clusters.
- **Training time:** MERF's EM loop multiplies training time by the number of
  iterations (15-30x). May need to reduce grid search scope or use warm-starting.
- **Library maturity:** `merf` v1.0 is a small open-source project. May need to
  vendor or extend for production use. Alternative: implement a simple EM loop
  manually (the algorithm is straightforward for random intercepts).
- **Interaction with per-target models:** Each target stat gets a separate GBM.
  MERF would need to be applied per-target, meaning separate random effects per
  stat per player. This is correct (a player may overperform on AVG but not SLG)
  but multiplies the stored state.

### Go/No-Go Gate

- **Go:** MERF improves batter RMSE on >= 4/6 targets vs the best feature-only
  model (post-Phases 2-4). Mean batter RMSE improvement >= 3%. No pitcher target
  degrades > 2%.
- **Partial go:** Improves 2-3 targets. Apply MERF selectively to targets where
  it helps (e.g., AVG and BABIP but not ISO).
- **No-go:** Feature engineering from Phases 2-4 already captures sufficient
  player-level signal, and the architectural complexity of MERF doesn't justify
  the marginal improvement. Accept the feature-only model.

---

## Summary

| Phase | Focus | Status | Result |
|-------|-------|--------|--------|
| 1 | Residual persistence diagnostic | **Complete** | **NO-GO** — 1/6 persistence, 0/6 ceiling |
| 2 | Spray-by-batted-ball-type features | Proposed (High) | Reframed: ISO signal + 400+ PA bucket |
| 3 | Multi-year rolling features | Proposed (Medium) | Reframed: denoising, not persistence |
| 4 | Career experience features | Proposed (Low) | Weakened justification |
| 5 | MERF (player-level random intercepts) | Deprioritized | No persistent residuals to model |

### Phase ordering rationale (updated post-Phase 1)

Phase 1 diagnostic returned NO-GO: model residuals are near-random, meaning
the model is already capturing the persistent skill components that research
identified in raw stat gaps. The "batter persistence gap" framing is retired.

Remaining phases are reordered by standalone merit:

1. **Phase 2 (spray-by-batted-ball-type)** remains highest priority. ISO is the
   only stat with persistent residuals (r = 0.106), and spray-batted-ball
   interactions are the known mechanism. The 400+ PA bucket also shows elevated
   correlations for power stats (slg: 0.169, iso: 0.190), suggesting these
   features could help where it matters most.

2. **Phase 3 (multi-year rolling features)** remains viable for denoising
   single-season Statcast inputs, though the "multi-year memory" framing is
   weakened. Evaluate on RMSE improvement independent of persistence.

3. **Phase 4 (career experience)** is downgraded. The model is not
   systematically under-regressing veterans, so the experience-as-prior
   justification is weak. Consider only after Phases 2-3 are evaluated.

4. **Phase 5 (MERF)** is deprioritized indefinitely. Player-level random
   intercepts learn persistent residual patterns, but residuals are not
   persistent. MERF would fit noise. Revisit only if Phases 2-4 reveal
   residual structure that feature engineering alone cannot capture.

### Expected impact (revised)

The Phase 1 ceiling analysis showed **no exploitable residual persistence gap**.
Naively applying prior residuals makes predictions worse (-16% to -39% RMSE).
The model's remaining batter error is dominated by genuinely unpredictable
noise (BABIP variance, in-game matchup effects, sequencing luck).

Phases 2-3 may still yield incremental RMSE improvements through better input
features, but expectations should be modest — the model is already performing
near the ceiling for Statcast-based batter projections. Improvements are more
likely to come from the live model roadmap's feature engineering (sprint speed,
park factors, platoon splits) than from persistence-focused work.
