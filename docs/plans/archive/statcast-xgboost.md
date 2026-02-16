# Statcast xGBoost Over/Underperformance Model

## Goal

Build a gradient-boosted model that uses Statcast quality-of-contact and pitch-tracking
metrics (barrel%, exit velocity, hard-hit%, launch angle, spin rate, chase rate, etc.)
to predict expected performance. By comparing predicted ("expected") stats to actual
results, the model identifies players who over- or underperformed their underlying skill
indicators — surfacing regression candidates and breakout signals.

## Motivation

Traditional projection systems (Marcel, PECOTA, Steamer) rely heavily on rate-stat
history and regression to the mean. Statcast metrics offer a skills-based lens:

- A batter with elite barrel% and exit velocity who posted a low BABIP likely
  underperformed and is due for positive regression.
- A pitcher whose swinging-strike rate and spin profile declined but whose ERA stayed
  low is a regression-to-worse candidate.

A gradient-boosted model is well-suited here because:

- It handles non-linear feature interactions (e.g., exit velo × launch angle sweet spot).
- It naturally produces feature importance, making over/underperformance *explainable*.
- It is robust to missing features and doesn't require feature scaling.

## Architecture

The model plugs into the existing protocol system:

- Implements `Model`, `Preparable`, `Trainable`, `Evaluable`, `Predictable`, `Ablatable`,
  `FeatureIntrospectable`.
- Uses `DatasetAssembler` for feature materialization (SQL + Statcast transforms).
- Stores trained artifacts as serialized model files (`ArtifactType.FILE`).
- Produces projections storable in the `projection` table.
- Over/underperformance deltas surface via the evaluation pipeline and a new CLI report.

## Phases

### Phase 1: Statcast CLI Ingest

The Statcast data infrastructure exists (schema, repo, `StatcastSource`, feature DSL)
but there is no CLI command to load data. Add one.

**Deliverables:**

- `fbm ingest statcast --season <years>` command.
- Wire `StatcastSource` → `StatsLoader` → `StatcastPitchRepo` using the existing
  loader pattern.
- Add a column mapper that transforms the pybaseball Statcast DataFrame into
  `StatcastPitch` domain entities.
- Support incremental loads (the `load_log` table tracks what has been loaded).

### Phase 2: Extended Statcast Feature Transforms

The existing `batted_ball` and `pitch_mix` transforms cover exit velocity, launch angle,
barrel%, and pitch-type usage. Add transforms for the remaining skill indicators the
model needs.

**New transforms:**

| Transform | Source columns | Outputs |
|-----------|---------------|---------|
| Plate discipline | `zone`, `description`, `plate_x`, `plate_z` | `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct` |
| Expected stats | `estimated_ba_using_speedangle`, `estimated_woba_using_speedangle` | `xba`, `xwoba`, `xslg` (estimated from xwOBA scaling) |
| Spin profile | `release_spin_rate`, `pitch_type`, `pfx_x`, `pfx_z` | `avg_spin_rate`, `ff_spin`, `sl_spin`, `cu_spin`, `avg_h_break`, `avg_v_break` |

Each transform follows the existing `TransformFeature` pattern: a pure function grouped
by `(player_id, season)`, registered as a module-level constant.

**Deliverables:**

- `plate_discipline.py`, `expected_stats.py`, `spin_profile.py` in
  `features/transforms/`.
- Unit tests for each transform function with edge cases (zero pitches, missing data).

### Phase 3: Add scikit-learn Dependency and Model Scaffold

The project currently has no ML library. Add scikit-learn (which includes the
`GradientBoostingRegressor` / `HistGradientBoostingRegressor`).

**Deliverables:**

- Add `scikit-learn>=1.6.0` to `[project.dependencies]`.
- Create `src/fantasy_baseball_manager/models/xgboost/` package:
  - `__init__.py`
  - `model.py` — the model class, registered as `"statcast-gbm"`.
  - `features.py` — declares feature sets (Statcast transforms + traditional lags).
  - `targets.py` — defines target stats for batters and pitchers.
  - `serialization.py` — save/load trained model artifacts (joblib).
- The model class implements `Model`, `Preparable`, `Trainable`, `Evaluable`,
  `Predictable`, `Ablatable`, `FeatureIntrospectable`.
- Stub implementations that pass protocol conformance tests.
- Use `HistGradientBoostingRegressor` (supports missing values natively, faster than
  the classic `GradientBoostingRegressor`).

### Phase 4: Batter Training and Evaluation Pipeline

Implement the full train → evaluate → predict loop for batters.

**Feature set (batters):**

- Statcast transforms: `batted_ball`, `plate_discipline`, `expected_stats`.
- Traditional lags: PA, HR, H, 2B, 3B, BB, SO, SB (1- and 2-year lags).
- Age.

**Target stats:**

- Rate stats: batting average (AVG), on-base percentage (OBP), slugging (SLG),
  wOBA, ISO, BABIP, HR/FB.
- One model per target stat (multi-output via separate regressors).

**Train:**

- Materialize feature set via `DatasetAssembler`.
- Split by season (train on seasons `[:-1]`, evaluate on final season).
- Fit one `HistGradientBoostingRegressor` per target stat.
- Serialize fitted models to `artifacts/<model_name>/<version>/`.
- Record `TrainResult` metrics (train RMSE, feature count).

**Evaluate:**

- Load held-out season.
- Score predictions vs actuals: RMSE, MAE, R-squared per target stat.
- Return `EvalResult` with per-stat metrics.

**Predict:**

- Load trained models from artifact path.
- Materialize features for the prediction season.
- Generate predictions; return as `PredictResult` for storage in `projection` table.

**Deliverables:**

- Working `fbm train statcast-gbm`, `fbm evaluate statcast-gbm`,
  `fbm predict statcast-gbm` for batters.
- Integration tests using a small synthetic dataset.

### Phase 5: Pitcher Training and Evaluation Pipeline

Same structure as Phase 4, adapted for pitchers.

**Feature set (pitchers):**

- Statcast transforms: `pitch_mix`, `spin_profile`, `plate_discipline` (from the
  pitcher's perspective: batter_id → pitcher_id grouping).
- Traditional lags: IP, K, BB, HR, ERA, FIP (1- and 2-year lags).
- Age.

**Target stats:**

- Rate stats: ERA, FIP, K/9, BB/9, HR/9, BABIP, LOB%, WHIP.

**Deliverables:**

- Player-type-aware feature declarations (`player_type="pitcher"` filtering).
- Train/evaluate/predict for pitchers using the same model class.
- Tests.

### Phase 6: Over/Underperformance Report

With trained models producing expected stats, compute and surface the delta between
expected and actual performance.

**Deliverables:**

- `fbm report overperformers --season <year> --player-type <batter|pitcher>` command.
- `fbm report underperformers --season <year> --player-type <batter|pitcher>` command.
- Report shows: player name, actual stat, expected stat, delta, percentile rank of delta.
- Backed by a service that loads predictions and actuals from repos, computes deltas.
- Sorted by magnitude of over/underperformance.
- Rich-formatted table output.

### Phase 7: Ablation and Feature Importance

Leverage the `Ablatable` protocol to provide feature-importance analysis.

**Deliverables:**

- `ablate()` implementation using permutation importance on the held-out set.
- `fbm ablate statcast-gbm` outputs a ranked feature-importance table.
- Useful for understanding which Statcast metrics drive over/underperformance
  identification.

### Phase 8: Ensemble Integration

Wire the Statcast GBM model into the existing ensemble framework as a component
projection system, so its expected-stat predictions blend with Marcel, Steamer, etc.

**Deliverables:**

- Ensure `statcast-gbm` predictions are stored with a system name usable by the
  ensemble model.
- Test blending `statcast-gbm` with Marcel in a weighted-average ensemble.
- Document recommended ensemble weights in configuration.

## Dependencies

| Phase | Depends on |
|-------|-----------|
| 1     | — (existing infra) |
| 2     | Phase 1 (needs Statcast data loaded for integration tests) |
| 3     | — (can proceed in parallel with 1-2) |
| 4     | Phases 1, 2, 3 |
| 5     | Phase 4 (reuses model class and patterns) |
| 6     | Phase 4 (needs trained model + predictions) |
| 7     | Phase 4 |
| 8     | Phase 4, Phase 5 |

## Open Questions

- **Minimum PA/IP threshold**: What qualifying thresholds to require for training
  samples? Suggest 200 PA for batters, 50 IP for pitchers to start.
- **Hyperparameter tuning**: Phase 4 uses default `HistGradientBoostingRegressor`
  params. A future phase could add cross-validated tuning.
- **Sprint speed**: Statcast publishes sprint-speed data separately from pitch-level
  data. Consider a future phase to ingest and incorporate it.
- **Aging curve interaction**: Should the GBM model include aging features, or leave
  aging adjustments to the ensemble layer?
