# Test Infrastructure & Coverage Roadmap

**Created:** 2026-02-17
**Status:** Proposed
**Goal:** Add coverage measurement, test markers, and fill identified coverage
gaps to strengthen the existing test suite.

## Motivation

The project has 2,218 tests with strong patterns (Protocol-based fakes,
constructor injection, parallel execution), but lacks tooling to measure what
those tests actually cover. There are no test markers for selective execution,
and a handful of modules have no dedicated test files. Adding these capabilities
will catch regressions, speed up local development, and surface untested paths.

## Constraints

- Do not break existing test parallelism (`pytest-xdist`) or randomization
  (`pytest-randomly`).
- Coverage thresholds should be achievable immediately — set based on current
  measured coverage, then ratchet up.
- Test markers should be opt-in — existing tests continue to run by default.
- New tests follow project conventions: constructor-injected fakes, no
  `monkeypatch` for domain/service logic, TDD workflow.

## Phase 1 — Coverage Measurement

Add `pytest-cov` and establish a baseline.

- Add `pytest-cov` to dev dependencies: `uv add --group dev pytest-cov`.
- Add coverage configuration to `pyproject.toml`:
  - `[tool.coverage.run]` — source = `["src"]`, omit test files.
  - `[tool.coverage.report]` — show missing lines, set `fail_under` to the
    measured baseline (round down to nearest 5).
- Add `--cov` to pytest addopts or document as a manual invocation
  (`uv run pytest --cov`). Do not add to default addopts if it conflicts with
  xdist performance.
- Run coverage once, record baseline percentage, set `fail_under` accordingly.
- Add `.coveragerc` or equivalent to `.gitignore` for the `.coverage` data file.

## Phase 2 — Test Markers

Add markers for selective test execution.

- Define markers in `pyproject.toml` under `[tool.pytest.ini_options]`:
  - `integration` — tests that touch SQLite (real connections, migrations).
  - `slow` — tests that take >1s (feature materialization, full model runs).
- Apply `@pytest.mark.integration` to repo tests and feature integration tests.
- Apply `@pytest.mark.slow` to full-pipeline tests (CLI end-to-end, statcast
  materialization).
- Document marker usage in CLAUDE.md:
  - Fast local run: `uv run pytest -m "not slow"`.
  - Unit only: `uv run pytest -m "not integration"`.
- Verify: existing `uv run pytest` (no marker filter) still runs everything.

## Phase 3 — Fill Coverage Gaps

Write tests for modules that currently lack dedicated test files.

- `services/dataset_catalog.py` — add `tests/services/test_dataset_catalog.py`
  covering catalog listing, rebuild, and edge cases.
- `models/gbm_training.py` — add `tests/models/test_gbm_training.py` covering
  shared GBM training utilities (hyperparameter handling, cross-validation
  helpers).
- `models/distributions.py` — add `tests/models/test_distributions.py` covering
  distributional projection logic (percentile computation, distribution families).
- `models/ensemble/` — expand test coverage for ensemble weighting, missing
  system handling, and edge cases.
- After writing new tests, re-run coverage and ratchet `fail_under` up if the
  new baseline exceeds the previous threshold.

## Phase 4 — Pre-commit Coverage Gate (Optional)

Add coverage enforcement to the pre-commit quality gate.

- Evaluate whether adding `--cov --cov-fail-under=N` to the pre-commit pytest
  hook is practical (may slow commits if coverage collection is expensive with
  xdist).
- If too slow for pre-commit, document as a CI-only gate and add to a future CI
  pipeline configuration.

## Success Criteria

- `uv run pytest --cov` produces a coverage report with per-module breakdown.
- `fail_under` threshold is set and enforced (either pre-commit or CI).
- `pytest -m "not slow"` runs in under 30s for fast local iteration.
- All identified coverage gap modules have dedicated test files.
- No existing tests are broken by marker additions.
