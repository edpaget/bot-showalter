# Marcel Derived Features — Roadmap

## Context

The Marcel engine currently does two significant data derivations internally in Python:

1. **Weighted-average rates** — `weighted_average_rates()` combines 3 years of lagged stats with configurable weights (5/4/3) into per-PA/IP rates. `_weighted_playing_time()` does the same for the regression denominator.
2. **League averages** — `compute_league_averages()` aggregates all players' most-recent-season stats into league-wide rates used for regression.

Both are general-purpose data transformations that happen to live inside a specific model. They should be expressible as features so that:

- Other models can reuse the same weighted rates and league context without reimplementing them.
- The feature set version hash captures the weights/parameters, so changing them triggers correct re-materialization.
- The engine becomes a thinner pipeline: regression, age adjustment, and PT projection — all pure model logic.

### Design approach

The existing `TransformFeature` operates on **raw source data** (statcast pitches, batting_stats rows). The transforms we need here operate on **already-materialized lag columns** in the dataset table. A new `DerivedTransformFeature` type handles this distinction cleanly:

- Same `(group_by, transform, outputs)` pattern as `TransformFeature`.
- Declares `inputs` (column names from the materialized table) instead of `source` + `columns`.
- Assembler runs derived transforms **after** the SQL pass and source-transform pass, as a third pass.

---

## Phase 1 — DerivedTransformFeature infrastructure

Add the new feature type and wire it through the assembler.

### 1a. `DerivedTransformFeature` type

**File:** `features/types.py`

```python
@dataclass(frozen=True)
class DerivedTransformFeature:
    name: str
    inputs: tuple[str, ...]        # columns to read from materialized table
    group_by: tuple[str, ...]      # e.g. ("player_id", "season") or ("season",)
    transform: RowTransform         # same protocol as TransformFeature
    outputs: tuple[str, ...]        # new columns to add
    version: str | None = None
```

Update `AnyFeature` union:

```python
type AnyFeature = Feature | DeltaFeature | TransformFeature | DerivedTransformFeature
```

Add to `_feature_to_dict` for version hashing. Add to `features/__init__.py` exports.

### 1b. Assembler derived-transform pass

**File:** `features/assembler.py`

After the existing source-transform pass in `materialize()`, add a third pass:

1. `ALTER TABLE` to add output columns (same pattern as source transforms).
2. `SELECT {inputs} FROM [{table_name}]` — reads from the dataset table itself.
3. Group by the declared keys, apply the transform, `UPDATE` results back.

The query is simpler than source transforms — no JOINs to external tables needed.

### 1c. SQL generator skip

**File:** `features/sql.py`

`_extract_features` already skips `TransformFeature`. Extend to skip `DerivedTransformFeature`.

### 1d. Tests

- Unit test the assembler derived-transform pass with a trivial transform (e.g., sum two columns).
- Test that derived transforms see columns produced by the SQL pass.
- Test that derived transforms see columns produced by source transforms (ordering guarantee).
- Test version hashing includes derived transform definitions.

---

## Phase 2 — Weighted-average-rates transform

Replace `weighted_average_rates()` and `_weighted_playing_time()` with a derived feature.

### 2a. Transform function

**File:** `features/transforms/weighted_rates.py` (new)

A factory function parameterized by categories, weights, and PT column:

```python
def make_weighted_rates_transform(
    categories: Sequence[str],
    weights: tuple[float, ...],
    pt_column: str,              # "pa" or "ip"
) -> Callable[[list[dict]], dict[str, Any]]:
```

The returned transform:
- Reads `{cat}_1, {cat}_2, ..., {cat}_N` and `{pt}_1, ..., {pt}_N` from a single-row group.
- Computes `rate[cat] = sum(stat_i * w_i) / sum(pt_i * w_i)` — same formula as `engine.weighted_average_rates()`.
- Also outputs `weighted_pt = sum(pt_i * w_i)` (needed later as the regression denominator).
- Returns `{"{cat}_wavg": rate, ..., "weighted_pt": value}`.

### 2b. Feature factory

**File:** `models/marcel/features.py`

```python
def build_batting_weighted_rates(
    categories: Sequence[str],
    weights: tuple[float, ...],
) -> DerivedTransformFeature:
```

Constructs the `DerivedTransformFeature` with:
- `inputs`: all lag columns for categories + PA (`hr_1`, `hr_2`, `hr_3`, `pa_1`, `pa_2`, `pa_3`, ...).
- `group_by`: `("player_id", "season")` — each group is a single row.
- `outputs`: `("{cat}_wavg", ..., "weighted_pt")`.
- `version`: derived from the weights tuple for cache correctness.

Analogous `build_pitching_weighted_rates` using IP.

### 2c. Tests

- Test the transform function directly against the existing `weighted_average_rates()` with identical inputs — results must match.
- Test `weighted_pt` output matches `_weighted_playing_time()`.
- Test edge cases: zero PT seasons, fewer seasons than weights, all-zero stats.
- Integration test: materialize a feature set that includes the weighted-rates derived transform, verify output columns exist and are correct.

---

## Phase 3 — League-average transform

Replace `compute_league_averages()` with a derived feature.

### 3a. Transform function

**File:** `features/transforms/league_averages.py` (new)

```python
def make_league_avg_transform(
    categories: Sequence[str],
    pt_column: str,
) -> Callable[[list[dict]], dict[str, Any]]:
```

The returned transform:
- Receives **all rows for a season** (grouped by `(season,)`).
- Reads `{cat}_1` and `{pt}_1` from each row (most-recent-year raw stats — same as what `compute_league_averages()` uses via `seasons[0]`).
- Computes `league_rate[cat] = sum({cat}_1 across players) / sum({pt}_1 across players)`.
- Returns `{"league_{cat}_rate": rate, ...}`.

### 3b. Feature factory

**File:** `models/marcel/features.py`

```python
def build_batting_league_averages(
    categories: Sequence[str],
) -> DerivedTransformFeature:
```

Constructs the `DerivedTransformFeature` with:
- `inputs`: lag-1 columns for categories + PA.
- `group_by`: `("season",)` — cross-player aggregation.
- `outputs`: `("league_{cat}_rate", ...)`.

The assembler's UPDATE-by-group-key pattern handles the broadcast naturally: `UPDATE table SET league_hr_rate = ? WHERE season = ?` writes the league rate to every row in that season.

Analogous `build_pitching_league_averages` using IP.

### 3c. Tests

- Test the transform function against `compute_league_averages()` with identical player data — results must match.
- Test with zero-PT players excluded correctly.
- Integration test: verify every row in a season gets the same league-average values.

---

## Phase 4 — Simplify Marcel engine

Update the engine to consume pre-computed features instead of computing them internally.

### 4a. Update convert.py

**File:** `models/marcel/convert.py`

Replace `rows_to_player_seasons()` with a simpler extraction that reads the flat derived columns:

```python
def rows_to_marcel_inputs(
    rows: list[dict],
    categories: Sequence[str],
    pitcher: bool = False,
) -> dict[int, MarcelInput]:
```

Where `MarcelInput` carries:
- `weighted_rates: dict[str, float]` — from `{cat}_wavg` columns.
- `weighted_pt: float` — from `weighted_pt` column.
- `league_rates: dict[str, float]` — from `league_{cat}_rate` columns.
- `age: int`
- Per-lag season lines (still needed for `project_playing_time` which uses raw PT per season).

### 4b. Simplify engine.py

**File:** `models/marcel/engine.py`

- Remove `weighted_average_rates()`, `_weighted_playing_time()`, `compute_league_averages()`.
- `project_player()` receives pre-computed weighted rates, weighted PT, and league averages.
- Pipeline becomes: regress → age-adjust → project PT → multiply. Three steps instead of five.

### 4c. Update model.py

**File:** `models/marcel/model.py`

- `_build_feature_sets()` includes the derived transforms (weighted rates + league averages) parameterized from `MarcelConfig`.
- `predict()` no longer calls `compute_league_averages()` — reads league rates from materialized rows.
- `predict()` passes pre-extracted rates/league data to the simplified engine.

### 4d. Tests

- All existing `test_engine.py` tests must still pass (adapted to the new signatures).
- End-to-end test: same input data produces identical projections before and after the refactor.
- Verify that changing weights in config produces a different feature set version (triggers re-materialization).

---

## Phase order and dependencies

```
Phase 1 (DerivedTransformFeature infra)
  ↓
Phase 2 (Weighted rates)  ←→  Phase 3 (League averages)   [independent of each other]
  ↓                              ↓
Phase 4 (Simplify engine)  [depends on both 2 and 3]
```

Phases 2 and 3 can be done in either order or in parallel — both depend only on Phase 1. Phase 4 depends on both being complete.

---

## Out of scope

- **Regression to the mean as a derived feature.** Regression uses a model-specific parameter (`regression_n`) and is more "model logic" than "data derivation." Could be a Phase 5 follow-up if the pattern proves useful, but the current scope keeps it in the engine.
- **Age adjustment and PT projection.** These are pure model decisions (peak age, baselines, role detection) — not data features.
- **Extending the SQL generator with weighted-window syntax.** The `DerivedTransformFeature` approach reuses existing assembler patterns rather than complicating the SQL layer.
