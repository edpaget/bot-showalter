# Composite GBM Engine

Replace the composite model's fixed Marcel regression with a learned GBM engine that actually consumes all materialized feature columns. The feature infrastructure (SQL assembler, feature groups, dataset materialization) is already in place — what's missing is an engine that reads the full feature matrix instead of only the Marcel-specific columns.

## Current state

All composite variants (composite, composite-mle, composite-statcast, composite-full) run the same Marcel regression engine:

```
weighted_rates → regress_to_mean(league_averages) → age_adjust → rate × projected_PT
```

The engine only reads `{cat}_wavg`, `weighted_pt`, and `league_{cat}_rate` columns via `rows_to_marcel_inputs()`. Extra feature columns (statcast-gbm rates, MLE rates, batted ball metrics, plate discipline) are materialized into the dataset table by the SQL assembler but never consumed. All variants produce identical predictions.

## Target architecture

```
fbm.toml                    feature group registry         composite model
┌──────────────────┐        ┌──────────────────────┐       ┌────────────────────┐
│ [models.comp-X]  │        │ static groups        │       │ _resolve_group()   │
│ feature_groups = │───────>│   age                │──────>│ _build_feature_sets│
│   [...]          │        │   projected_*_pt     │       │                    │
│ engine = "gbm"   │        │   mle_batter_rates   │       │ compose_feature_set│
└──────────────────┘        │   statcast_gbm_*     │       │                    │
                            │   statcast_batted_*  │       │ GBM engine         │
                            │   ...                │       │   train per-target │
                            └──────────────────────┘       │   predict rates    │
                                                           │   rate × PT        │
                                                           └────────────────────┘
```

When `engine = "gbm"` is set in model_params, the composite model trains one `HistGradientBoostingRegressor` per target stat on all feature columns, then predicts rate stats and derives counting stats via rate × projected PT. When `engine` is absent or `"marcel"`, behavior is unchanged.

## Design decisions

**Predict rates, derive counting stats.** The GBM predicts rate stats (avg, obp, slg, era, whip, etc.) — the same targets as statcast-gbm. Counting stats are derived by multiplying rates by projected PT. This matches how steamer/zips/statcast-gbm work and avoids the GBM having to learn PT (which is handled by the playing-time model).

**Reuse statcast-gbm training infrastructure.** The `training.py` module (`extract_features`, `extract_targets`, `fit_models`, `score_predictions`, `compute_permutation_importance`) is generic — it operates on row dicts and column name lists. Move it to a shared location so both statcast-gbm and composite can use it.

**One model per target.** Each rate stat gets its own GBM. This is the same approach as statcast-gbm and works well because targets have different valid-row subsets (e.g., batters with 0 AB can't have AVG).

**Training feature set = prediction feature set + target columns at lag 0.** Following the statcast-gbm pattern: training sets include `target_{stat}` columns at lag 0, prediction sets do not.

**Feature columns are extracted dynamically** from the FeatureSet, not hardcoded. This is critical — different composite variants have different feature groups, so the column list must adapt.

## Phases

### Phase 1: Extract shared training utilities ✅

Move the generic GBM training functions out of `statcast_gbm/training.py` into a shared module that both models can import.

**Files:**
- Create `src/fantasy_baseball_manager/models/training.py` — move `TargetVector`, `extract_features`, `extract_targets`, `fit_models`, `score_predictions`, `compute_permutation_importance` (and the private `_filter_X` helper)
- Update `src/fantasy_baseball_manager/models/statcast_gbm/model.py` — import from new location
- Update `src/fantasy_baseball_manager/models/statcast_gbm/training.py` — re-export from shared module for backwards compatibility, or delete if no other consumers

**Verification:** All existing statcast-gbm tests pass unchanged.

### Phase 2: Composite training feature sets ✅

Add the ability to build training-mode feature sets for composite variants. Training sets need target columns (lag-0 rate stats) appended to the feature columns.

**Files:**
- `src/fantasy_baseball_manager/models/composite/features.py` — add `composite_target_features()` for batter and pitcher targets, `build_training_feature_sets()` that takes the prediction FeatureSets from `_build_feature_sets()` and appends target features
- `src/fantasy_baseball_manager/models/composite/targets.py` — define `BATTER_TARGETS` and `PITCHER_TARGETS` tuples (same stats as statcast-gbm, possibly extended)
- Tests for feature set construction

**Targets:**
- Batters: avg, obp, slg, woba, iso, babip (same as statcast-gbm)
- Pitchers: era, fip, k_per_9, bb_per_9, hr_per_9, babip, whip (same as statcast-gbm)

**Design note:** The composite model's `_build_feature_sets()` already produces the right prediction-time FeatureSets. The training variant just needs target columns added. A helper like `append_targets(feature_set, target_features) → FeatureSet` can do this generically.

### Phase 3: Composite feature column extraction ✅

Add a function to extract the ordered list of feature column names from a composite FeatureSet. This is the bridge between the feature system and the GBM training loop — the GBM needs a flat `list[str]` of column names to build its feature matrix.

**Files:**
- `src/fantasy_baseball_manager/models/composite/features.py` — add `feature_columns(fs: FeatureSet) → list[str]` that walks the FeatureSet's features and collects column names (handling Feature, TransformFeature outputs, DerivedTransformFeature outputs). Must exclude target columns.
- Tests verifying column extraction matches materialized dataset columns

**Key concern:** DerivedTransformFeature outputs (like `h_wavg`, `league_h_rate`) need correct naming. Verify by materializing a dataset and comparing `row.keys()` against the extracted column list.

### Phase 4: Engine protocol and routing ✅

Define a `CompositeEngine` protocol and wrap the existing Marcel logic in a `MarcelEngine`. Add engine selection based on `model_params["engine"]`.

**Files:**
- `src/fantasy_baseball_manager/models/composite/engine.py` — define `CompositeEngine` protocol with `train()` and `predict()` methods, implement `MarcelEngine` wrapping current logic
- `src/fantasy_baseball_manager/models/composite/model.py` — route through engine based on config; update `supported_operations` to include `"train"`, `"evaluate"`, `"ablate"` when engine is `"gbm"`

**Protocol shape:**
```python
class CompositeEngine(Protocol):
    def predict(
        self,
        rows: list[dict[str, Any]],
        feature_set: FeatureSet,
        pt: dict[int, float],
        config: EngineConfig,
    ) -> list[CompositePlayerPrediction]: ...
```

The MarcelEngine wraps the existing `rows_to_marcel_inputs → regress_to_mean → age_adjust → rate × PT` pipeline. No behavior change — just extraction into an engine interface.

**Verification:** All existing composite tests pass. Marcel engine produces identical output.

### Phase 5: GBM engine — train ✅

Implement `GBMEngine.train()` that fits per-target GBMs on the composite feature matrix.

**Files:**
- `src/fantasy_baseball_manager/models/composite/engine.py` — add `GBMEngine` class with `train()` method
- `src/fantasy_baseball_manager/models/composite/model.py` — wire up `train()` operation, season splitting (train on N-1 seasons, holdout on last), artifact serialization

**Train flow:**
1. Build training feature sets (prediction features + target columns)
2. Materialize via assembler
3. Split by season (train / holdout)
4. Extract feature matrix X and target vectors y using shared training utilities
5. Fit one HistGradientBoostingRegressor per target
6. Score on holdout, report RMSE per target
7. Serialize models to `artifacts/composite-{variant}/{version}/`

**GBM hyperparameters** passed via `model_params`:
- `max_iter` (default 100)
- `max_depth` (default 5)
- `learning_rate` (default 0.1)
- `min_samples_leaf` (default 20)

### Phase 6: GBM engine — predict

Implement `GBMEngine.predict()` that loads trained models and produces projections.

**Files:**
- `src/fantasy_baseball_manager/models/composite/engine.py` — add `predict()` to `GBMEngine`
- `src/fantasy_baseball_manager/models/composite/convert.py` — may need a new converter or adjustments to handle GBM output format

**Predict flow:**
1. Build prediction feature sets (no target columns)
2. Materialize via assembler
3. Load trained models from artifacts
4. Extract feature matrix X
5. Predict rates per target
6. Look up projected PT per player
7. Derive counting stats: `counting[stat] = rate[stat] × PT`
8. Derive additional rate stats (ops = obp + slg, etc.)
9. Convert to domain Projections via `composite_projection_to_domain`

**Counting stat derivation from rates:**
- Batters: `h = avg × ab`, `ab = pa - bb - hbp - sf`, `hr = iso × ab` (approximate), etc. This is the inverse of `_compute_batter_rates`. May need a dedicated `rates_to_counting()` function. Alternatively, predict both rate and counting stats as separate targets and let the GBM learn them independently.
- Pitchers: `er = era × ip / 9`, `bb = bb_per_9 × ip / 9`, `so = k_per_9 × ip / 9`

**Alternative approach:** Predict counting stats directly (h, hr, bb, so, er, etc.) and derive rates from them, matching what Marcel does. This avoids the lossy rates→counting inversion. Worth benchmarking both approaches — the right answer depends on which target formulation the GBM learns better.

### Phase 7: Evaluate and ablate

Wire up the `evaluate` and `ablate` operations for the GBM engine.

**Files:**
- `src/fantasy_baseball_manager/models/composite/model.py` — add `evaluate()` delegating to `ProjectionEvaluator`, add `ablate()` using permutation importance from shared training utilities
- Update `supported_operations` to advertise these capabilities when engine is GBM

**Evaluate:** Same as statcast-gbm — calls `evaluator.evaluate(system, version, season)` against fangraphs actuals.

**Ablate:** Same as statcast-gbm — trains on N-1 seasons, measures permutation importance per feature on holdout season. This directly answers "does adding statcast-gbm rates improve composite predictions?"

### Phase 8: Variant benchmarking

With the GBM engine working, run each composite variant and compare against baselines.

**Comparisons to run:**
```bash
# Train all variants
fbm train composite --season 2022 2023 2024
fbm train composite-mle --season 2022 2023 2024
fbm train composite-statcast --season 2022 2023 2024
fbm train composite-full --season 2022 2023 2024

# Predict and compare
fbm predict composite --season 2025
fbm predict composite-mle --season 2025
fbm predict composite-statcast --season 2025
fbm predict composite-full --season 2025
fbm compare composite/latest composite-mle/latest composite-statcast/latest composite-full/latest steamer/2025 zips/2025 --season 2025 --top 300

# Ablation — which features matter?
fbm ablate composite-full --season 2022 2023 2024
```

**Success criteria:**
- At least one composite-gbm variant beats Marcel on rate-stat RMSE
- composite-statcast rate-stat RMSE is within 2x of steamer/zips (down from 3-7x)
- Feature ablation shows statcast-gbm and/or MLE features have positive importance
- If no improvement: the ablation data reveals which features are noise vs signal, guiding further work

**No-code phase** — this is analysis and comparison only.

## Risk and open questions

1. **Counting vs rate targets.** Should the GBM predict rates (avg, era) or counting stats (h, er)? Rates are the natural formulation but the GBM may overfit to low-PT players. Counting-stat targets with rate derivation (what Marcel does) may be more robust. Phase 6 should benchmark both.

2. **Feature sparsity.** Projection-derived features (MLE rates, statcast-gbm rates) are NULL when upstream models haven't run for a given season. HistGradientBoostingRegressor handles NaN natively, so this should work, but verify that sparse features don't hurt.

3. **Training data volume.** With 3 seasons of data (~1500-2000 players/season), we have ~5000 training rows. This is adequate for HistGradientBoostingRegressor but may benefit from regularization (low max_depth, high min_samples_leaf).

4. **Circular dependencies.** composite-statcast uses statcast-gbm predictions as features. If composite-statcast is also predicting the same targets, there's a risk of leakage if statcast-gbm was trained on overlapping data. The lag structure (statcast-gbm at lag=1) should prevent this, but verify.

5. **Marcel engine preservation.** The Marcel engine must remain the default. Existing behavior should not change for users who don't set `engine = "gbm"`.
