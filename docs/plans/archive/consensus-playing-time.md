# Consensus Playing Time — Roadmap

## Goal

Use the average of preseason ZiPS and Steamer PA/IP projections as a **consensus playing time** signal, serving two purposes:

1. **Feature input for the playing time model.** Consensus PT encodes depth chart expectations, injury history discounting, and roster-slot judgment that our historical features alone cannot capture. The Hardball Times regression model achieved R² = 0.74 largely because its "starter status" variable encoded exactly this kind of information.

2. **Normalization for rate-model evaluation.** When evaluating whether Marcel, statcast-gbm, or the ensemble produce accurate *rate* projections, playing time error confounds the signal. By substituting consensus PT for our own PT estimates, we isolate rate-prediction accuracy from playing-time accuracy.

## Background

We already have historical Steamer and ZiPS projections loaded (or loadable) into the `projection` table via the existing `import` command. The feature DSL already supports `Source.PROJECTION` with `.system()` filtering — projection columns can be joined into feature datasets just like batting or pitching stats. The infrastructure exists; this roadmap adds the features and wiring.

## Phase 1 — Consensus PT Feature

**Goal:** Compute `consensus_pa` (batters) and `consensus_ip` (pitchers) as the average of Steamer and ZiPS preseason projections, available as a feature in any feature set.

### 1a. Raw projection features

Add projection-sourced features that pull PA/IP from each system for the target season (lag 0):

```python
# In a new module: features/consensus_pt.py

from fantasy_baseball_manager.features.types import DerivedTransformFeature, Feature, Source

def _steamer_pa() -> Feature:
    return Feature(
        name="steamer_pa",
        source=Source.PROJECTION,
        column="pa",
        lag=0,
        system="steamer",
    )

def _zips_pa() -> Feature:
    return Feature(
        name="zips_pa",
        source=Source.PROJECTION,
        column="pa",
        lag=0,
        system="zips",
    )
```

These join against the `projection` table filtered by `system = 'steamer'` / `system = 'zips'` for the same `(player_id, season)` as the spine row. The SQL generation already handles this via `_plan_joins` and `_join_clause`.

### 1b. Derived consensus feature

A `DerivedTransformFeature` that averages the two (or uses whichever is available when only one system covers a player):

```python
def make_consensus_pt_transform(stat: str) -> RowTransform:
    """Average steamer_{stat} and zips_{stat}, falling back to whichever is present."""
    steamer_key = f"steamer_{stat}"
    zips_key = f"zips_{stat}"
    def transform(rows: list[dict[str, Any]]) -> dict[str, Any]:
        row = rows[0]
        s = row.get(steamer_key)
        z = row.get(zips_key)
        if s is not None and z is not None:
            return {f"consensus_{stat}": (s + z) / 2}
        return {f"consensus_{stat}": s if s is not None else z}
    return transform
```

### 1c. Tests

- `consensus_pa` is the mean of Steamer and ZiPS PA when both are present.
- Falls back to the single available system when only one covers the player.
- Returns `None` when neither system covers the player.
- Feature SQL generation produces the correct joins for two distinct projection systems in the same feature set.

**Exit criteria:** `consensus_pa` and `consensus_ip` are usable features that can be included in any `FeatureSet`.

## Phase 2 — Playing Time Model Integration

**Goal:** Add `consensus_pa` / `consensus_ip` as a feature in the playing time model's feature set and measure the impact.

### 2a. Feature registration

Add consensus PT features to `build_batting_pt_features()` and `build_pitching_pt_features()` in `models/playing_time/features.py`. This adds them to the dataset but doesn't yet force the regression to use them.

### 2b. Ablation study

Run the existing ablation study with `consensus_pa` / `consensus_ip` as an additional feature group. This will show the marginal lift from adding consensus PT on top of the existing features (age, prior PA, WAR, etc.).

Expected outcome: consensus PT should be the single most predictive feature in the model — it already encodes most of what our other features try to approximate, plus depth chart information we don't have.

### 2c. Evaluate standalone consensus baseline

Compare three approaches on holdout data:

| Approach | Description |
|----------|-------------|
| Marcel native | `0.5 * PA_1 + 0.1 * PA_2 + 200` |
| Our PT model (current) | Ridge regression on historical features |
| Consensus PT alone | Raw average of Steamer + ZiPS PA |
| Our PT model + consensus | Ridge regression with consensus as an additional feature |

This answers: does our model add anything *on top of* what ZiPS/Steamer already provide?

### 2d. Tests

- Playing time model can train and predict with consensus features.
- Ablation output includes the consensus feature group.
- Model handles missing consensus values (players not covered by ZiPS/Steamer).

**Exit criteria:** Quantified comparison of PT model accuracy with and without consensus PT.

## Phase 3 — Rate-Model Evaluation with Consensus PT

**Goal:** Evaluate rate-prediction accuracy of Marcel, statcast-gbm, and the ensemble when playing time error is removed.

### 3a. Evaluation mode

Add an option to the `evaluate` command (or a new subcommand) that substitutes consensus PT for the system's own PT projection when computing counting stats:

```
fbm evaluate marcel/2025-pre --season 2025 --normalize-pt consensus
```

This would:
1. Take Marcel's per-PA rate projections (AVG, OBP, SLG, HR/PA, etc.)
2. Multiply by consensus PA (instead of Marcel's own PA)
3. Compare the resulting counting stats against actuals

### 3b. Rate-only metrics

Add evaluation metrics that only measure rate stats (AVG, OBP, SLG, ERA, WHIP, K/9, etc.) regardless of PT. This is simpler than PT substitution — just compare rates directly without weighting by PA.

### 3c. Comparative report

Produce a side-by-side report showing for each system:

| System | Rate RMSE (AVG) | Rate RMSE (OBP) | Counting RMSE (HR) | Counting RMSE (HR, consensus PT) |
|--------|-----------------|------------------|---------------------|-----------------------------------|
| Marcel | ... | ... | ... | ... |
| statcast-gbm | ... | ... | ... | ... |
| ensemble | ... | ... | ... | ... |

This isolates: "which system predicts rates best?" from "which system predicts playing time best?"

### 3d. Tests

- PT normalization correctly substitutes consensus PA/IP.
- Rate-only metrics exclude PT-dependent stats.
- Report renders for systems with and without consensus coverage.

**Exit criteria:** Clear quantitative answer to "are our rate projections good, independent of PT accuracy?"

## Phase 4 — Marcel Consensus PT Mode

**Goal:** Let Marcel use consensus PT instead of its native formula, as a configurable option.

### 4a. Playing time source abstraction

Marcel already reads PT projections from the DB when available. Add a config-driven option:

```toml
[models.marcel.params]
playing_time = "consensus"  # "native" | "consensus" | "playing-time-model"
```

When `playing_time = "consensus"`, Marcel pulls the average Steamer/ZiPS PA/IP for each player and uses that as the PT multiplier for its rate projections.

### 4b. Fallback chain

For players not covered by consensus (no Steamer or ZiPS projection): fall back to Marcel's native formula. Log the fallback count so we know how many players are affected.

### 4c. Tests

- Marcel uses consensus PT when configured.
- Fallback to native works for uncovered players.
- Config option is respected.

**Exit criteria:** `fbm predict marcel --season 2026` can use consensus PT.

## Phase 5 — Ensemble with Consensus PT

**Goal:** Use consensus PT as the playing time foundation for the ensemble model.

### 5a. PT source for blending

The ensemble engine's `blend_rates` function already separates rate-averaging from PT-averaging. Wire it to use consensus PT instead of averaging the component systems' PT projections.

### 5b. Config

```toml
[models.ensemble.params]
playing_time = "consensus"
```

### 5c. Tests

- Ensemble uses consensus PT for final counting-stat projections.
- Rate blending still uses component weights.
- Fallback behavior for uncovered players.

**Exit criteria:** Ensemble projections use consensus PT as the playing time foundation.

## Data Requirements

For each season in the evaluation window, we need preseason Steamer and ZiPS projections loaded:

```
fbm import steamer data/steamer-bat-{year}.csv --version {year}-pre --player-type batter --season {year}
fbm import steamer data/steamer-pit-{year}.csv --version {year}-pre --player-type pitcher --season {year}
fbm import zips data/zips-bat-{year}.csv --version {year}-pre --player-type batter --season {year}
fbm import zips data/zips-pit-{year}.csv --version {year}-pre --player-type pitcher --season {year}
```

Steamer is available 2012–2025; ZiPS is available 2010–2025. The intersection (2012–2025) gives 14 seasons of consensus PT data for backtesting.

## Out of Scope

- **Building our own depth chart model.** The whole point is to consume expert consensus rather than replicate it.
- **In-season PT updates.** This is preseason consensus only.
- **Weighting Steamer vs. ZiPS differently.** Start with a simple average; optimize weights only if the ablation shows it matters.
- **Other projection systems (ATC, THE BAT).** Start with the two systems with the longest historical archives. Add others later if needed.
