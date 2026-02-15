# Model Composition — Roadmap

## Context

The system currently supports a single model producing projections end-to-end (Marcel: raw stats → weighted rates → regression → age adjustment → counting stats). Two composition patterns would expand this:

1. **Ensembles** — Combine outputs from multiple projection systems via weighted average. Useful when several models capture different signal: Marcel is conservative, a regression model might be more reactive, third-party imports add external signal.

2. **Stacked models** — One model's output feeds another. The motivating case: a playing-time model predicts PA/IP, and a separate rate model predicts per-PA/IP rates. A composite step multiplies them to produce counting stats. This separates the "how much will they play" question from "how well will they hit per PA" question, letting each use different methods and data.

Both patterns should work within the existing `Model` protocol — no new abstractions needed for the model interface itself. The main infrastructure work is making prior projections available as input.

### Design constraints

- **Projections use flat columns** (`pa`, `hr`, `bb`, etc.), same schema as `batting_stats`. The `SqliteProjectionRepo` maps between flat columns and the domain `Projection.stat_json` dict. The SQL generator uses direct column references (`alias.hr`) — no `json_extract` needed.
- **`Feature` already carries `system` and `version` fields.** The `FeatureBuilder` DSL supports `.system()` and `.version()`. The SQL join planner filters projection joins by system/version/player_type. Direct column access, join planning, and filtering all work end-to-end.
- **Ensemble models don't need the feature system at all.** They operate on already-stored projections, not on raw data. A model that depends on `ProjectionRepo` is sufficient.

---

## Phase 1 — Projection feature validation and test coverage

`Source.PROJECTION` features already work with flat columns: direct access, join planning, system/version/player_type filtering are all tested in `TestProjectionRoundTrip` (6 integration tests). The `projection` `SourceRef` is already exported from `features/__init__.py`.

This phase adds missing validation and test coverage.

### 1a. Validation: require `.system()` for projection features

**File:** `features/types.py`

`FeatureBuilder.alias()` now raises `ValueError` when `source == Source.PROJECTION` and `system` is `None`. A projection feature without a system filter is ambiguous and previously caused a `KeyError` at SQL generation time.

### 1b. Rate stat coverage for projections

Added `pa` to all projection seed data fixtures and added tests for projection features with `denominator` (e.g., `hr / pa`). The generic `_raw_expr` handles denominators correctly for projections — just needed test coverage.

### 1c. Tests added

- `TestFeatureBuilder.test_projection_without_system_raises` — validates the new error.
- `TestFeatureBuilder.test_projection_with_system_succeeds` — confirms valid usage still works.
- `TestSelectExprProjection.test_projection_rate` — unit test for rate SQL expression.
- `TestProjectionRoundTrip.test_projection_rate_stat` — integration test with real DB.

---

## Phase 2 — Ensemble model ✅

A model that reads projections from multiple systems and produces a weighted-average projection.

### 2a. Ensemble engine

**File:** `models/ensemble/engine.py` (new)

Pure functions for combining projections:

```python
def weighted_average(
    projections: list[tuple[dict[str, Any], float]],  # (stat_json, weight) pairs
    stats: Sequence[str],
) -> dict[str, float]:
    """Weighted average of stat values across systems."""
```

- For each stat, compute `sum(value_i * weight_i) / sum(weight_i)` across systems that have the stat.
- Handle missing stats gracefully: if a system doesn't project a stat, exclude it from the average (and adjust the weight denominator).
- Rates (in `stat_json["rates"]`) and counting stats are averaged separately. Rates get weight-averaged directly; counting stats get weight-averaged directly (not recomputed from averaged rates).

Also support a `blend` mode where rates are averaged and then multiplied by averaged PA/IP:

```python
def blend_rates(
    projections: list[tuple[dict[str, Any], float]],
    rate_stats: Sequence[str],
    pt_stat: str,  # "pa" or "ip"
) -> dict[str, float]:
    """Average rates across systems, average PT, recompute counting stats."""
```

### 2b. Ensemble model class

**File:** `models/ensemble/model.py` (new)

```python
class EnsembleModel:
    name = "ensemble"
    description = "Weighted-average ensemble of multiple projection systems"
    supported_operations = frozenset({"predict"})
    artifact_type = "none"
```

Constructor takes `ProjectionRepo` (injected by factory). No `DatasetAssembler` needed.

`predict()` reads `model_params`:

```python
{
    "components": {"marcel": 0.6, "steamer": 0.4},
    "mode": "weighted_average",   # or "blend_rates"
    "season": 2025,
    "stats": ["h", "hr", "r", "rbi", "bb", "so", "sb"],   # optional filter
    "pt_stat": "pa"               # for blend_rates mode
}
```

Flow:
1. For each component system, fetch projections via `projection_repo.get_by_season(season, system=system)`.
2. Group by `(player_id, player_type)`.
3. For each player present in at least one system, apply `weighted_average` or `blend_rates`.
4. Emit `Projection` with `system="ensemble"`.

Players missing from a component system: use only the systems that have them, re-normalizing weights.

### 2c. Registration and factory wiring

**File:** `models/ensemble/__init__.py` — register with `@register("ensemble")`.

**File:** `cli/factory.py` — `build_model_context` needs to provide `projection_repo` kwarg when creating the model, since `create_model` already forwards matching kwargs:

```python
model = create_model(
    model_name,
    assembler=assembler,
    projection_repo=SqliteProjectionRepo(conn),
)
```

This is backwards-compatible — `create_model` filters kwargs by constructor signature, so Marcel (which doesn't accept `projection_repo`) ignores it.

### 2d. Tests

- Unit test `weighted_average` with known inputs — verify arithmetic.
- Test missing-system handling: player in system A but not B uses only A's projection.
- Test `blend_rates` mode: verify rates are averaged, counting stats = rate * PT.
- Test model `predict()` end-to-end with an in-memory SQLite DB seeded with projections from two systems.
- Test that ensemble projections are stored with `system="ensemble"`.

---

## Phase 3 — Stacked playing-time model ✅

Demonstrate the stacked pattern: a playing-time model produces PA/IP projections, and a rate model consumes them via `Source.PROJECTION` features to compute counting stats.

### 3a. Playing-time model

**File:** `models/playing_time/model.py` (new)

A focused model that only projects PA (batters) and IP (pitchers). Uses the feature system and `DatasetAssembler` for inputs.

Implementation approach: weighted historical PT with regression to a baseline, plus aging. This is essentially Marcel's `project_playing_time()` extracted as a standalone model. The engine logic already exists — this phase wraps it as a separate `Model` whose `stat_json` contains only `{"pa": N}` or `{"ip": N}`.

Supported operations: `prepare`, `predict`.

Constructor takes `DatasetAssembler`.

Feature set: lagged PA/IP columns + age (same features Marcel uses for PT projection, minus the rate columns).

### 3b. Composite rate-to-counting model

**File:** `models/composite/model.py` (new)

A model that:
1. Declares `Source.PROJECTION` features to read PA/IP from the playing-time model's output.
2. Declares raw batting/pitching features for rate computation.
3. Combines them: `counting_stat = rate * projected_pt`.

This model uses `DatasetAssembler` — the projection features come through the feature system (Phase 1 infra). Its `predict()` reads the materialized dataset where each row has both the playing-time projection and the historical rate features, then applies rate estimation + scaling.

Feature set example:

```python
features = (
    projection.col("pa").system("playing_time").lag(0).alias("proj_pa"),
    batting.col("h").per("pa").lag(1).alias("h_rate_1"),
    batting.col("h").per("pa").lag(2).alias("h_rate_2"),
    batting.col("h").per("pa").lag(3).alias("h_rate_3"),
    player.age(),
    # ... weighted rates derived transform on the rate columns ...
    # ... league averages derived transform ...
)
```

The engine reads `proj_pa` from the materialized row and multiplies by regressed/aged rates — the same pipeline as Marcel but with PT coming from an external model instead of being computed internally.

### 3c. CLI orchestration

Running stacked models requires ordering: the playing-time model must `predict` before the composite model can `prepare`/`predict` (since the composite model's features read from the playing-time projections table).

Options:
1. **Manual ordering** — user runs `fbm predict playing_time` then `fbm predict composite`. Simple, explicit.
2. **Pipeline config** — `model_params` declares dependencies, and the CLI runs them in order.

Start with manual ordering (option 1). A pipeline abstraction is premature until there are enough stacked models to justify it.

Document the dependency in the model's description and in `fbm predict --help` output for the composite model.

### 3d. Tests

- Unit test the playing-time model produces `stat_json` with only `pa` or `ip`.
- Integration test the full stack: seed raw stats → run playing-time predict → run composite predict → verify composite projections have counting stats that reflect the PT model's output.
- Test that changing the playing-time model's version triggers re-materialization of the composite model's feature set (because the projection feature's version is part of the feature set hash).

---

## Phase 4 — Evaluation support for composed models

Ensure the evaluation and comparison infrastructure works cleanly with ensembles and stacked models.

### 4a. Evaluator: per-system comparison

`ProjectionEvaluator.compare()` already accepts a list of `(system, version)` pairs. Verify that ensemble and composite systems appear correctly in comparison output alongside their component systems. No code change expected — just test coverage.

### 4b. CLI: projection lineage display

**File:** `cli/_output.py`

When displaying ensemble or composite projections via `fbm projections lookup`, include a note about the source systems. The ensemble model should store component metadata in `stat_json`:

```python
stat_json = {
    "h": 162,
    "hr": 42,
    "_components": {"marcel": 0.6, "steamer": 0.4},
    "_mode": "weighted_average"
}
```

The lookup display can show "ensemble (marcel 60%, steamer 40%)" as a subtitle.

For the composite model, store `_pt_system` in `stat_json` so the display can show "composite (PT: playing_time v2024.1)".

### 4c. Tests

- Test `compare()` output includes ensemble and composite systems.
- Test projection lookup display formatting for ensemble and composite projections.

---

## Phase order and dependencies

```
Phase 1 (Source.PROJECTION validation + coverage) ✅
  ↓
Phase 2 (Ensemble model) ✅        Phase 3 (Stacked PT model) ✅
  [independent of each other, but both depend on Phase 1]
  ↓                                ↓
Phase 4 (Evaluation support)  [depends on 2 and 3]
```

Phase 2 technically doesn't need Phase 1 (it uses `ProjectionRepo` directly, not the feature system), so it could be done in parallel with Phase 1. But Phase 1 is a small, self-contained change that benefits both patterns, so doing it first keeps things clean.

Phases 2 and 3 are independent of each other and can be done in either order.

---

## Out of scope

- **Pipeline/DAG orchestration** — Automatic dependency resolution and ordered execution of model chains. Premature until there are enough stacked models to justify the abstraction. Manual ordering is sufficient.
- **Per-stat system selection** — Picking the best system per stat (e.g., Marcel for HR, Steamer for SB) rather than weighting. Interesting but adds complexity; can be added as another ensemble mode later.
- **Bayesian model averaging** — Weighting systems by their historical accuracy. Requires evaluation history infrastructure. A natural follow-up once evaluation data accumulates.
- **Dynamic weight tuning** — Optimizing ensemble weights against historical actuals. Requires a training step for the ensemble model. Can be added by making `EnsembleModel` implement `Trainable`.
