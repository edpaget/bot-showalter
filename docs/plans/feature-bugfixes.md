# Feature System Bug Fixes — Roadmap (Complete)

All 4 phases implemented. 9 bugs fixed across CLI transaction discipline, RunManager correctness, projection SQL filtering, and TransformFeature type safety.

## Phase 1 — CLI transaction discipline ✅

Fixed missing commits and connection leaks in the CLI layer.

- **Bug 1:** `delete_run` never committed DB deletion after removing disk artifacts
- **Bug 2:** `finalize_run` upsert was never committed — model run records silently lost
- **Bug 3:** CLI commands created connections that were never closed

Commits: `a413cb8`, `2c6ee0b`

## Phase 2 — RunManager correctness ✅

Fixed logic errors in RunManager and ProjectionEvaluator.

- **Bug 4:** `finalize_run` hardcoded `artifact_type=none` instead of propagating the model's declared type
- **Bug 5:** Tautological `if projections:` guard inside a `for proj in projections:` loop

Commit: `d4c627f`

## Phase 3 — Projection feature SQL ✅

Fixed under-filtered projection JOINs that produced indeterminate results.

- **Bug 6:** Projection joins lacked `version` filtering — ambiguous when multiple versions exist
- **Bug 7:** Projection joins lacked `player_type` filtering — ambiguous for two-way players

Commit: `9eea9a3`

## Phase 4 — TransformFeature type safety ✅

Completed TransformFeature integration into downstream consumers.

- **Bug 8:** `print_features` had no `TransformFeature` branch (already fixed during Phase 5 of feature DSL)
- **Bug 9:** `RowTransform` not exported from `features/__init__.py`; added test coverage for `print_features` across all `AnyFeature` variants

Commit: `5a00d8a`

## Non-issues investigated

The following were flagged during audit but confirmed as non-bugs:

- **`_extract_features` and `generate_sql` handling of `TransformFeature`** — Both correctly skip `TransformFeature` instances.
- **`Source.STATCAST` missing from `_SOURCE_TABLES`** — Expected; STATCAST is Phase 6 material for `TransformFeature`.
- **`_create_dataset` atomicity** — `get_or_materialize` compensates by checking table existence. Low risk.
- **`FeatureBuilder` mutable state** — Fluent pattern via `SourceRef.col()` creates fresh builders each time.
- **Column names interpolated in SQL** — Constructed in application code, not user input. Safe.
