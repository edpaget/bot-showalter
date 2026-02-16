# Statcast GBM: Three Operating Modes

## Context

The statcast-gbm model currently operates as a same-season true-talent estimator.
This roadmap describes three distinct modes that would make it useful across the
full fantasy baseball calendar: pre-season projections, in-season monitoring, and
feeding adjustments back into Marcel.

## The Three Modes

### Mode 1: Pre-Season Projection (Lagged Statcast Features)

**Use case:** Draft-day projections. Use last year's Statcast data to predict next
year's rate stats, just like Marcel uses last year's counting stats.

**What changes:** The Statcast transform features (batted ball, plate discipline,
expected stats, pitch mix, spin profile) currently join on same-season data:

```sql
-- Current: same-season join
AND CAST(SUBSTR(sc.game_date, 1, 4) AS INTEGER) = d.season

-- Lagged: prior-season join
AND CAST(SUBSTR(sc.game_date, 1, 4) AS INTEGER) = d.season - 1
```

**Implementation:**

1. Add `lag: int = 0` field to `TransformFeature` in `features/types.py`.

2. In `assembler.py:_build_raw_query`, apply the lag offset:
   ```python
   season_expr = "d.season" if tf.lag == 0 else f"d.season - {tf.lag}"
   # ... AND CAST(SUBSTR(sc.game_date, 1, 4) AS INTEGER) = {season_expr}
   ```

3. Add a `lag()` method to `TransformFeature` (or a builder) that returns a
   copy with the lag set:
   ```python
   BATTED_BALL_LAG1 = BATTED_BALL.with_lag(1)
   ```

4. Create a second feature set builder in `statcast_gbm/features.py`:
   ```python
   def build_batter_preseason_set(seasons):
       features = [player.age()]
       features.extend(_batter_lag_features())
       features.extend([
           BATTED_BALL.with_lag(1),
           PLATE_DISCIPLINE.with_lag(1),
           EXPECTED_STATS.with_lag(1),
       ])
       return FeatureSet(name="statcast_gbm_batting_preseason", ...)
   ```

5. The model's `predict()` selects which feature set based on a config param
   (e.g., `mode: "preseason"` vs `mode: "true_talent"`).

6. Retrain a separate model artifact for the preseason feature set — different
   model weights because the features have different predictive relationships
   when lagged.

**Training data structure:**
- Row: player X, season 2024
- Lag features: 2023 PA, 2022 PA (same as today)
- Statcast features: 2023 exit velo, 2023 whiff rate (lagged by 1)
- Targets: 2024 AVG, OBP, SLG (same as today)

This means training needs 3+ seasons (e.g., 2022-2024) so that the earliest
training row (2023) has prior-year Statcast data (2022) available.

**What we'd learn:** How much predictive power the Statcast features retain when
lagged one year. The ablation study showed lag stats contribute ~0 importance in
the current model — with lagged Statcast features, the model would rely entirely
on those lagged transforms, and we'd see whether exit velo / whiff rate from the
prior year actually predicts next year's rates better than Marcel's regression.

---

### Mode 2: In-Season True-Talent Estimator (Over/Under Monitor)

**Use case:** Mid-season roster management. Compare a player's statcast-gbm
true-talent estimate to their actual stats to identify who's running hot/cold
relative to their underlying quality.

**What changes:** This mode already mostly works — the current model produces
true-talent estimates from same-season Statcast data. What's missing is a
dedicated report that compares those estimates to actual stats and surfaces
the biggest divergences.

**Implementation:**

1. Add a new CLI command `fbm report talent-delta` (or extend the existing
   `report` command) that:
   - Loads statcast-gbm projections for the current season
   - Loads FanGraphs actuals for the same season
   - Computes per-player deltas: `actual_stat - statcast_estimate`
   - Ranks players by divergence magnitude
   - Outputs overperformers (positive delta on rate stats like AVG) and
     underperformers (negative delta)

2. The report uses the existing `ProjectionComparison` / `PlayerStatDelta`
   domain models from `domain/projection_accuracy.py` and
   `domain/performance_delta.py`. No new domain types needed.

3. Add interpretive context to the output:
   ```
   Overperformers (AVG actual >> statcast estimate — regression candidates)
   Player          Actual AVG  xAVG (statcast)  Delta
   J. Soto         .310        .285              +.025
   ...

   Underperformers (AVG actual << statcast estimate — buy-low candidates)
   Player          Actual AVG  xAVG (statcast)  Delta
   M. Trout        .220        .258              -.038
   ```

4. Optionally support partial-season Statcast ingestion (e.g., first-half only)
   so the model can be run at the trade deadline with data through July.

**Key insight:** The statcast-gbm estimate reflects what a player *should* be
doing based on quality of contact and plate discipline. Large gaps between the
estimate and actual stats suggest BABIP luck, sequencing effects, or defensive
positioning — factors likely to regress. This is the classic "buy low / sell
high" signal for fantasy.

---

### Mode 3: Statcast-Adjusted Marcel (Feed Adjustments Into Projections)

**Use case:** Better pre-season projections. Use the gap between a player's
statcast true-talent estimate and their actual stats to adjust Marcel's
projection for the following season.

**Concept:** If statcast-gbm says a player's true talent was .280 AVG but he
hit .250, Marcel would normally project based on the .250 actual. A
statcast-adjusted Marcel would partially credit the .280 estimate, projecting
something closer to .265.

**Implementation:**

This follows the existing MLE augmentation pattern in `marcel/mle_augment.py`,
which already blends external rate estimates into Marcel's weighted rates.

1. Create `models/marcel/statcast_augment.py` (mirrors `mle_augment.py`):

   ```python
   def augment_with_statcast(
       inputs: list[MarcelInput],
       statcast_projections: list[Projection],
       blend_weight: float = 0.3,
   ) -> list[MarcelInput]:
       """Blend statcast true-talent rates into Marcel's weighted rates."""
   ```

   For each player with both Marcel inputs and a statcast-gbm projection:
   - Extract the statcast rate estimates (avg, obp, slg, era, etc.)
   - Blend them into `weighted_rates` using a configurable weight:
     ```python
     blended_rate = marcel_rate * (1 - w) + statcast_rate * w
     ```
   - Return modified `MarcelInput` with the blended rates

2. The blend happens **before** regression to the mean. This means:
   - Marcel still regresses the blended rates toward league average
   - The statcast signal gets dampened for small-sample players (appropriate)
   - Age adjustment still applies after regression (appropriate)

3. Wire it into `marcel/model.py` predict flow, gated on config:
   ```toml
   [models.marcel.params]
   statcast_augment = true
   statcast_system = "statcast-gbm"
   statcast_version = "latest"
   statcast_weight = 0.3
   ```

4. The weight parameter controls how much to trust the statcast estimate vs
   the raw actual stats. A weight of 0.3 means "30% statcast true-talent,
   70% actual results" — a moderate adjustment.

**Example:**
- Player hit .250 in 2024 (Marcel sees this)
- Statcast-gbm estimates .280 true talent for 2024 (same-season model)
- Blended rate: .250 * 0.7 + .280 * 0.3 = .259
- Marcel regresses .259 toward league average (.252) → ~.256
- Age-adjusts → final projection

Without the adjustment, Marcel would regress .250 → ~.251, producing a lower
projection. The statcast adjustment partially corrects for the bad-luck season.

**Why this is better than a simple ensemble:** The ensemble (Mode 1 of the
current system) blends Marcel's *output* with statcast-gbm's *output*. The
statcast-adjusted Marcel blends statcast signal into Marcel's *input*, letting
Marcel's regression, age adjustment, and playing-time machinery still operate
on it. This preserves Marcel's strengths (calibrated regression, counting stats)
while improving the rate-stat inputs.

## Relationship Between Modes

The three modes serve different points in the calendar:

```
           Pre-season         In-season          Post-season
           (Feb-Mar)          (Apr-Sep)          (Oct-Jan)

Mode 1:    [==========]
           Lagged statcast
           pre-season proj

Mode 2:                       [================]
                              True-talent delta
                              buy low/sell high

Mode 3:    [==========]                          [==========]
           Adjusted Marcel                       Compute adjustments
           uses prior year's                     from completed season
           Mode 2 deltas                         for next year's Marcel
```

Mode 2 feeds Mode 3: the in-season true-talent gaps from the completed season
become the adjustment factors for next year's Marcel projection.

## Implementation Order

**Phase 1: Mode 2 — In-Season Monitor** (smallest change, immediate value)
- New CLI report command
- Uses existing statcast-gbm predictions + existing performance delta logic
- No model changes needed

**Phase 2: Mode 3 — Statcast-Adjusted Marcel** (moderate change, high value)
- New `statcast_augment.py` following existing MLE pattern
- Config-gated integration in Marcel predict flow
- Requires Mode 2 to have been run for the prior season

**Phase 3: Mode 1 — Pre-Season Lagged Model** (largest change, speculative value)
- Feature DSL extension (TransformFeature lag)
- Assembler SQL changes
- Separate model artifact with different weights
- Value is uncertain until we see how much signal degrades with lag
