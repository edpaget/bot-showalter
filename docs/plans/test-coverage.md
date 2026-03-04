# Test Coverage Roadmap

Raise test coverage from 89.1% toward 95%+ by (1) marking untestable code with `no cover` pragmas or coverage omissions, (2) writing tests for CLI output formatters (pure functions that are easy to test), (3) filling service-layer gaps, and (4) testing the small amount of real logic inside CLI commands. The current `fail_under` threshold is 88%; this roadmap will allow raising it significantly.

## Status

| Phase | Status |
|-------|--------|
| 1 — Pragma and omit housekeeping | done (2026-03-04) |
| 2 — CLI output formatter tests | done (2026-03-04) |
| 3 — Service-layer gap tests | done (2026-03-04) |
| 4 — CLI command logic tests | done (2026-03-04) |
| 5 — Raise fail_under threshold | not started |

## Phase 1: Pragma and omit housekeeping

Mark code that cannot or should not be unit-tested — protocol stubs, CLI command orchestration functions, DI wiring, Yahoo I/O, and interactive session methods.

### Context

The codebase has ~1,600 missed statements. A deep analysis shows the majority fall into categories that provide no value from unit testing: Typer command functions that wire services together (tested via the service layer already), DI factory functions, network I/O paths, and interactive REPL handlers. Excluding these lets us focus coverage on code with real logic.

### Steps

1. **Add `models/protocols.py` to the `omit` list** in `pyproject.toml` `[tool.coverage.run]`. It contains only Protocol stubs and frozen dataclasses — the 18 missed branches are all `...` method bodies.

2. **Mark CLI command orchestration functions `# pragma: no cover`** in files where the uncovered code is pure wiring (creates connections, calls services, prints output). Apply to the function signature line so the entire function is excluded. Target files:
   - `cli/commands/validate.py` — `preflight_cmd`, `full_cmd`, `_run_preflight`
   - `cli/commands/compare_features.py` — `compare_features_cmd`
   - `cli/commands/residuals.py` — `worst_misses`, `gaps`, `cohort`
   - `cli/commands/report.py` — all uncovered report subcommands (`report_talent_delta`, `report_talent_quality`, `report_residual_persistence`, `report_residual_analysis`, `report_value_over_adp`, `report_adp_accuracy`, `report_adp_movers`, `report_projection_confidence`, `report_variance_targets`, `report_system_disagreements`)
   - `cli/commands/datasets.py` — `datasets_list`, `datasets_drop`, `datasets_rebuild`
   - `cli/commands/model.py` — `model_train`, `model_evaluate`, `model_predict`, `model_tune`, `model_sweep`, `model_gate`
   - `cli/commands/ingest.py` — `ingest_statcast`, `ingest_sprint_speed`, `ingest_il`, `ingest_milb_batting`, `ingest_adp_bulk`, `_write_adp_csv`
   - `cli/commands/marginal_value.py` — `marginal_value_cmd`
   - `cli/commands/quick_eval.py` — `quick_eval_cmd`
   - `cli/commands/feature.py` — uncovered subcommands
   - `cli/commands/runs.py` — uncovered subcommands
   - `cli/commands/profile.py` — uncovered subcommands

3. **Mark `cli/factory.py` wiring functions `# pragma: no cover`**. These are pure DI constructors that assemble services from DB connections — no decision logic.

4. **Mark Yahoo I/O-bound code `# pragma: no cover`** where uncovered paths involve real network calls or OAuth flows:
   - `yahoo/draft_poller.py` — the polling loop (`run` method)
   - `yahoo/roster_source.py` — uncovered API call paths
   - `yahoo/transaction_source.py` — uncovered parsing edge cases
   - `yahoo/player_map.py` — network-dependent refresh path
   - `yahoo/auth.py` — OAuth redirect/refresh paths
   - `yahoo/client.py` — HTTP retry/error paths

5. **Mark `services/draft_session.py` interactive handlers `# pragma: no cover`**. The uncovered methods (`_handle_best`, `_handle_pool`, `_handle_balance`, `_handle_needs`, `_show_category_summary`, `_handle_report`, `_handle_save`, `_handle_quit`, display helpers) are interactive REPL handlers that call already-tested services and format output.

6. Run `uv run pytest --cov` and verify coverage increases.

### Acceptance criteria

- `models/protocols.py` is in the `omit` list in `pyproject.toml`.
- All newly-added `# pragma: no cover` annotations are on orchestration/wiring/I/O functions only — no business logic is excluded.
- `uv run pytest --cov` passes and shows a meaningful coverage increase (target: 93%+).
- No existing tests are removed or modified.

## Phase 2: CLI output formatter tests

Write tests for the pure functions in `cli/_output/` that transform domain objects into formatted strings or Rich renderables.

### Context

CLI output formatters are pure functions: domain dataclass in, string/table out. They have no side effects, no I/O, and no dependencies beyond their arguments. Despite being trivially testable, several have very low coverage: `_residuals.py` (5%), `_validate.py` (9%), `_draft.py` (64%), `_experiments.py` (83%), `_evaluation.py` (86%), `_keeper.py` (88%). Testing them catches regressions in user-facing output (wrong columns, missing fields, formatting errors).

### Steps

1. **`_residuals.py`** (5% → ~95%): Test `print_error_decomposition_report`, `print_feature_gap_report`, `print_cohort_bias_report`, `print_cohort_bias_summary`. Construct minimal domain objects (`ErrorDecompositionReport`, `FeatureGapReport`, `CohortBiasReport`) and assert key strings appear in captured output.

2. **`_validate.py`** (9% → ~95%): Test `print_preflight_result` and `print_validation_result`. Construct `PreflightResult` and `ValidationResult` domain objects and verify output contains expected labels, values, and color coding.

3. **`_draft.py`** (64% → ~95%): Test the uncovered formatters (`print_draft_board`, `print_draft_tiers`, `print_draft_report`, etc.). These take `DraftBoard`, `Tier`, `DraftReport` domain objects. Follow the pattern already used by the existing tests for this file.

4. **`_experiments.py`** (83% → ~95%): Test the uncovered sorting/filtering paths — experiment list sorting by delta, feature filtering by significance.

5. **`_evaluation.py`** (86% → ~95%): Test `_print_tail_accuracy_section` and `print_stratified_comparison_result`. Construct `SystemMetrics` and `StratifiedResult` domain objects.

6. **`_keeper.py`** (88% → ~95%): Test uncovered branches — likely edge cases in keeper decision formatting.

7. **`_profile.py`** (95%), **`_feature_factory.py`** (94%), **`_mock_draft.py`** (95%), **`_model.py`** (98%), **`_reports.py`** (97%)**: Fill remaining gaps where the effort is small (1-3 missed statements each).

### Acceptance criteria

- Every `cli/_output/` module is at 95%+ coverage.
- Tests construct domain objects directly (no DB, no monkeypatching).
- Tests use `capsys` or `io.StringIO` to capture output and assert on content.
- No `# pragma: no cover` is added to any output formatter function.

## Phase 3: Service-layer gap tests

Fill coverage gaps in services that contain real business logic.

### Context

Several services have uncovered branches representing real decision logic — not just trivial None checks but meaningful paths that affect correctness. These are the highest-value test targets in the codebase.

### Steps

1. **`services/performance_report.py`** (88% → 95%+): Lines 28, 40, 44, 51, 57, 61, 102, 150 are branches for different report section formatting and edge cases. Write tests that exercise the missing conditional paths.

2. **`services/regression_gate.py`** (88% → 95%+): Lines 154, 167-186 are the gate decision logic for edge cases (e.g., when regression is detected). Write tests with boundary inputs that trigger the uncovered gate paths.

3. **`services/residual_persistence_diagnostic.py`** (88% → 95%+): Lines 52, 80, 83, 86, 110, 114, 117 are conditional branches in the diagnostic analysis. Test with inputs that trigger low-sample, missing-data, and edge-case paths.

4. **`services/projection_evaluator.py`** (87% → 95%+): Lines 209-219, 228, 232, 235, 237, 248-250 contain evaluation logic for edge cases (empty results, missing stats, filtering). Write tests with sparse/empty input data.

5. **`services/draft_translation.py`** (88% → 95%+): Lines 68-74 are the fallback translation path. Write a test that triggers the fallback.

6. **`services/true_talent_evaluator.py`** (84% → 95%+): Lines 60, 85-88, 104, 107, 110, 133, 138, 141, 213 are None-guard branches for missing projections/actuals. Write tests with incomplete data that triggers each None path.

7. **`models/mle/engine.py`** (84% → 95%+): Test uncovered MLE computation paths.

8. **`services/player_biography.py`** (89% → 95%+): Lines 22, 56, 97, 112, 114, 128 are edge cases in biography lookup (missing data, unknown positions). Test with incomplete player records.

### Acceptance criteria

- Each listed service reaches 95%+ coverage.
- Tests use constructor-injected fakes/stubs (no monkeypatching services).
- All tests pass in isolation and in the full suite.

## Phase 4: CLI command logic tests

Test the small number of CLI command helper functions that contain real business logic (not orchestration).

### Context

While most CLI command code is pure orchestration (marked `no cover` in phase 1), a few helper functions contain real logic that warrants testing: type coercion, data merging, cohort dispatching, and feature injection.

### Steps

1. **`cli/commands/quick_eval.py` — `_coerce_value`**: This function coerces string parameters to bool/int/float with fallbacks. Write tests for each type path and the fallback-to-string case.

2. **`cli/commands/ingest.py` — `_build_player_teams`**: Merges Lahman roster stints with live MLB API data, with conflict resolution (API overrides Lahman). Test with overlapping/conflicting data, missing mlbam_ids, and empty inputs.

3. **`cli/commands/standalone.py` — `_build_cohort_assignments`**: Dispatches cohort assignment by dimension (age, experience, top300). Test each dimension path and verify the returned assignments.

4. **`cli/commands/residuals.py` — `_load_raw_features`**: Loads raw statcast features from SQLite result set. Test with a seeded in-memory DB.

5. **`cli/commands/model.py` — distribution construction** (lines 165-183): Builds `StatDistribution` objects from prediction dicts. Extract to a helper function if necessary, then test.

### Acceptance criteria

- Each helper function listed above has direct unit tests.
- Tests do not require monkeypatching Typer or the CLI runner — they call the helper functions directly.
- Coverage for these specific functions reaches 95%+.

## Phase 5: Raise fail_under threshold

Increase the coverage floor to lock in the gains from phases 1-4.

### Context

The current `fail_under` is 88%. After phases 1-4, actual coverage should be 95%+. Raising the floor prevents regressions.

### Steps

1. Run `uv run pytest --cov` and record the exact coverage percentage.
2. Set `fail_under` in `pyproject.toml` to 1% below the measured value (e.g., if coverage is 96.2%, set `fail_under = 95`).
3. Run the full suite to confirm the new threshold passes.
4. Update `CLAUDE.md` if needed to reflect the new coverage expectation.

### Acceptance criteria

- `fail_under` is set to at least 93 (likely higher).
- `uv run pytest --cov` passes with the new threshold.
- CI passes.

## Ordering

Phases should be done sequentially:

1. **Phase 1 first** — the pragma/omit changes are the lowest effort and produce the largest coverage jump. They also clarify what actually needs tests vs. what's excluded, making phases 2-4 easier to scope.
2. **Phase 2 next** — output formatters are pure functions with no dependencies, making them the easiest tests to write.
3. **Phase 3** — service-layer tests require constructing fakes but follow established patterns in the test suite.
4. **Phase 4** — CLI command logic tests are a small number of targeted tests for extracted helpers.
5. **Phase 5 last** — lock in the gains after all other phases are complete.

No phase depends on another to be mergeable — each phase independently improves coverage and can be shipped alone.
