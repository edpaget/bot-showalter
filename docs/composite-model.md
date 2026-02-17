# Composite Model

The composite model is a configurable projection system that assembles its feature set from named feature groups via TOML configuration. Different composite variants (composite-mle, composite-statcast, composite-full) select different feature combinations — no code changes needed to test a new combination.

## Design principles

- **Everything is a feature.** Cross-model signal (MLE translations, statcast-gbm rates, Marcel projections) flows through the feature system as `Source.PROJECTION` features. Every input is visible in the materialized dataset, individually inspectable, and ablatable.
- **Composition via configuration.** Each TOML `[models.<variant>.params]` section specifies a `feature_groups` list. The model class is shared; the variant name drives which features are materialized and how projections are labeled.
- **Marcel stays pure.** Marcel produces projections from raw stats alone. It does not consume other models' outputs. Cross-model signal enters downstream via composite variants.

## Architecture

```
fbm.toml                    feature group registry         composite model
┌──────────────────┐        ┌──────────────────────┐       ┌────────────────────┐
│ [models.comp-X]  │        │ static groups        │       │ _resolve_group()   │
│ feature_groups = │───────>│   age                │──────>│ _build_feature_sets│
│   ["age",        │        │   projected_*_pt     │       │                    │
│    "batting_...",│        │   mle_batter_rates   │       │ compose_feature_set│
│    "mle_..."]   │        │   statcast_gbm_*     │       │                    │
└──────────────────┘        │   statcast_batted_*  │       │ Marcel engine      │
                            │   ...                │       │   regress_to_mean  │
                            ├──────────────────────┤       │   age_adjust       │
                            │ parameterized groups │       │   rate × PT        │
                            │   batting_count_lags │       └────────────────────┘
                            │   pitching_count_lags│                │
                            │   batting_rate_lags  │                ▼
                            └──────────────────────┘         projection table
```

### Feature flow

1. **Config reads `feature_groups`** from `model_params` (or uses `DEFAULT_GROUPS`).
2. **`_resolve_group()`** maps each name to a `FeatureGroup` — static groups come from the registry, parameterized groups (counting lags, rate lags) are built by factory functions with categories/weights from `MarcelConfig`.
3. **Groups are split** by `player_type` into batter and pitcher lists. When counting lags are present, Marcel's `weighted_rates` and `league_averages` derived transforms are appended as a synthetic group.
4. **`compose_feature_set()`** merges groups into a single `FeatureSet`, deduplicating by feature name. The feature set version hash captures the full composition.
5. **`SqliteDatasetAssembler`** materializes the feature set into a `ds_{id}` table. SQL features (lags, projections) run in pass 1; Python transforms (batted ball, plate discipline) run in pass 2.
6. **Marcel regression engine** consumes counting lags, weighted rates, and league averages. Extra feature columns (MLE rates, statcast-gbm rates, batted ball metrics) are materialized but not consumed — they are staged for a future learned engine.

### Engine separation

The feature groups control *what data goes into the dataset*. The engine controls *how predictions are computed*. Currently all composite variants use the Marcel regression engine:

```
weighted_rates → regress_to_mean(league_averages) → age_adjust → rate × projected_PT
```

This means all variants currently produce identical predictions (since the engine only reads the counting/rate/league columns). The extra feature columns will differentiate variants when a learned engine (GBM) replaces the fixed-formula engine.

## Package structure

```
models/composite/
    __init__.py      # Registers aliases: composite-mle, composite-statcast, composite-full
    model.py         # CompositeModel, _resolve_group, DEFAULT_GROUPS
    convert.py       # composite_projection_to_domain (stamps system name on projections)
    features.py      # Legacy inline builders (kept for regression tests)

features/
    groups.py        # FeatureGroup dataclass, registry, compose_feature_set
    group_library.py # Static group registrations + factory functions
```

## Configuration

### Default groups

When no `feature_groups` key is present in `model_params`, `DEFAULT_GROUPS` is used:

```python
DEFAULT_GROUPS = (
    "age",
    "projected_batting_pt",
    "projected_pitching_pt",
    "batting_counting_lags",
    "pitching_counting_lags",
)
```

This produces the same features as the base composite model before the configurable architecture was added.

### TOML variant definitions

```toml
[models.composite-mle.params]
feature_groups = [
    "age", "projected_batting_pt", "projected_pitching_pt",
    "batting_counting_lags", "pitching_counting_lags",
    "mle_batter_rates",
]

[models.composite-statcast.params]
feature_groups = [
    "age", "projected_batting_pt", "projected_pitching_pt",
    "batting_counting_lags", "pitching_counting_lags",
    "statcast_gbm_batter_rates", "statcast_gbm_pitcher_rates",
    "statcast_batted_ball",
]

[models.composite-full.params]
feature_groups = [
    "age", "projected_batting_pt", "projected_pitching_pt",
    "batting_counting_lags", "pitching_counting_lags",
    "mle_batter_rates",
    "statcast_gbm_batter_rates", "statcast_gbm_pitcher_rates",
    "statcast_batted_ball", "statcast_plate_discipline",
]
```

### Adding a new variant

1. Add a TOML section with the desired `feature_groups`.
2. Add the alias name to `models/composite/__init__.py`:
   ```python
   for _alias in ("composite-mle", "composite-statcast", "composite-full", "composite-new"):
       register_alias(_alias, "composite")
   ```
3. Run `fbm predict composite-new --season 2022 2023 2024`.

No model code changes needed.

## Available feature groups

### Static groups (registered at import)

| Name | Player Type | Source | Contents |
|------|-------------|--------|----------|
| `age` | both | player | Player age |
| `positions` | both | player | Player positions |
| `projected_batting_pt` | batter | projection (playing_time) | Projected PA |
| `projected_pitching_pt` | pitcher | projection (playing_time) | Projected IP |
| `mle_batter_rates` | batter | projection (mle) | avg, obp, slg, iso, k_pct, bb_pct, babip, pa |
| `statcast_gbm_batter_rates` | batter | projection (statcast-gbm) | avg, obp, slg, woba, iso, babip |
| `statcast_gbm_pitcher_rates` | pitcher | projection (statcast-gbm) | era, fip, k/9, bb/9, hr/9, babip, whip |
| `marcel_batter_rates` | batter | projection (marcel) | avg, obp, slg, ops, pa |
| `marcel_pitcher_rates` | pitcher | projection (marcel) | era, whip, k/9, bb/9, ip |
| `statcast_batted_ball` | batter | statcast (transform) | exit_velo, launch_angle, barrel%, hard_hit%, gb/fb/ld%, sweet_spot%, ev_p90 |
| `statcast_plate_discipline` | both | statcast (transform) | chase_rate, zone_contact%, whiff_rate, swstr%, called_strike% |
| `statcast_expected_stats` | batter | statcast (transform) | xBA, xSLG, xwOBA |
| `statcast_pitch_mix` | pitcher | statcast (transform) | per-pitch-type usage% and velocity |
| `statcast_spin_profile` | pitcher | statcast (transform) | spin rates and movement profiles |

### Parameterized groups (built by factory)

| Name | Player Type | Parameters | Contents |
|------|-------------|------------|----------|
| `batting_counting_lags` | batter | categories, lags (from MarcelConfig weights) | PA + counting stats at each lag |
| `pitching_counting_lags` | pitcher | categories, lags (from MarcelConfig weights) | IP, G, GS + counting stats at each lag |
| `batting_rate_lags` | batter | columns, lags | Rate stats (avg, obp, slg, woba) at each lag |

Parameterized groups are resolved by `_resolve_group()` in `model.py` using `MarcelConfig` for categories and weight counts. They are not in the registry because their contents depend on config.

## Projection system naming

Each variant stamps its own name on projections via `system=self._model_name`:

```
composite      → projection.system = "composite"
composite-mle  → projection.system = "composite-mle"
composite-full → projection.system = "composite-full"
```

This allows independent evaluation and comparison:

```bash
fbm compare composite/latest composite-mle/latest composite-full/latest --season 2025
```

## Multi-name registration

The composite model class is registered once under `"composite"`. Variant names are aliases:

```python
# models/composite/__init__.py
for _alias in ("composite-mle", "composite-statcast", "composite-full"):
    register_alias(_alias, "composite")
```

The CLI factory passes the alias name to the constructor via `model_name`, so `CompositeModel(model_name="composite-mle").name` returns `"composite-mle"`.

## Dependencies and ordering

Running a composite variant requires upstream projections to exist:

```
playing_time model  ──→  projected_batting_pt / projected_pitching_pt features
mle model           ──→  mle_batter_rates features
statcast-gbm model  ──→  statcast_gbm_*_rates features
marcel model        ──→  marcel_*_rates features
```

Currently this ordering is manual:

```bash
fbm predict playing_time --season 2022 2023 2024
fbm predict marcel       --season 2022 2023 2024   # if using marcel_*_rates
fbm predict statcast-gbm --season 2022 2023 2024   # if using statcast_gbm_*
fbm predict composite-full --season 2022 2023 2024
```

Statcast transform features (batted_ball, plate_discipline) do not require upstream predictions — they are computed directly from pitch-level data in `statcast.db`.

## Current status

### What works

- Feature group registry with static and parameterized groups
- TOML-driven variant configuration
- Multi-name model registration with per-variant system names
- Dataset materialization with correct columns per variant
- Statcast transform features (batted_ball, plate_discipline) populate from pitch data
- Marcel regression engine produces projections from counting stats
- Independent evaluation and comparison of all variants

### What remains

- **Projection-derived features are sparse.** MLE and statcast-gbm rate columns are present but only populated when upstream models have run and stored projections for the matching season. Without upstream data, these columns are NULL.
- **All variants produce identical predictions.** The Marcel regression engine only consumes counting stats, weighted rates, and league averages. Extra feature columns are ignored. Variant differentiation requires a learned engine.

## Future: GBM engine

The next major step is adding `engine = "gbm"` as a model_params option. When selected, the composite model trains an XGBoost/LightGBM model on *all* feature columns (including projection features from other models), enabling:

- **Per-feature ablation.** The existing `ablate` operation measures per-feature RMSE deltas, answering "does MLE signal improve predictions?"
- **Automatic weighting.** The GBM learns optimal weights for different signal sources rather than using fixed regression formulas.
- **Variant differentiation.** Different feature groups produce different trained models with measurably different accuracy.

The feature infrastructure is in place. The engine swap is the remaining piece.
