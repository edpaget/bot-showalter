# Code Quality & CI Roadmap

**Created:** 2026-02-17
**Status:** Proposed
**Goal:** Clean up minor DI inconsistencies, improve internal code quality, and
establish a CI pipeline to enforce the quality gate on all branches.

## Motivation

The codebase has strong conventions enforced by pre-commit hooks, but a few
patterns have drifted from the ideal: optional-with-assertion dependencies,
repos re-instantiated on every property access, and no CI pipeline to catch
issues before merge. These are individually small but collectively worth
addressing as a quality pass.

## Constraints

- Each phase is independently shippable — no phase depends on another.
- Changes must pass the existing pre-commit quality gate (format, lint, type
  check, tests).
- CI pipeline should mirror the pre-commit gate exactly — no drift between
  local and CI checks.
- Prefer simplicity: GitHub Actions with a single job is sufficient.

## Phase 1 — Required Dependencies in Models

Eliminate `repo | None = None` patterns where the dependency is always required.

- Audit all model `__init__` signatures for `| None = None` parameters that are
  asserted non-None at usage time.
- For each such parameter:
  - Remove `| None` from the type annotation.
  - Remove the default `= None`.
  - Remove the `assert ... is not None` guard at usage sites.
  - Verify the composition root in `factory.py` already passes the dependency
    (it does in all current cases).
- Update corresponding tests to always provide the dependency.
- Models affected (verify each): `MLEModel`, `StatcastGBMModel`,
  `PlayingTimeModel`, `CompositeModel`, `EnsembleModel`.

## Phase 2 — IngestContainer Property Caching

Cache repo instances in `IngestContainer` to avoid re-instantiation.

- In `cli/factory.py`, change `IngestContainer` repo properties from:
  ```python
  @property
  def player_repo(self) -> SqlitePlayerRepo:
      return SqlitePlayerRepo(self._conn)
  ```
  to:
  ```python
  @functools.cached_property
  def player_repo(self) -> SqlitePlayerRepo:
      return SqlitePlayerRepo(self._conn)
  ```
- Apply to all repo/source properties on `IngestContainer`.
- Add `import functools` at the top of the module.
- Verify tests still pass — behavior is identical since repos are stateless
  wrappers, but this makes the single-instance intent explicit.

## Phase 3 — CI Pipeline

Add a GitHub Actions workflow that mirrors the pre-commit quality gate.

- Create `.github/workflows/ci.yml` with:
  - Trigger: push to `main`, pull requests targeting `main`.
  - Python 3.14 setup via `actions/setup-python`.
  - Install dependencies via `uv sync`.
  - Steps (matching pre-commit order):
    1. `uv run ruff format --check src tests`
    2. `uv run ruff check src tests`
    3. `uv run ty check src tests`
    4. `uv run pytest --tb=short -q`
  - Single job, no matrix (one Python version).
- Add branch protection rule recommendation in the PR description (require CI
  to pass before merge).

## Phase 4 — Architecture Documentation

Add a lightweight architecture overview to help future contributors.

- Create `docs/architecture.md` with:
  - Data flow diagram (text-based): ingestion → database → features → models →
    projections → valuations.
  - Layer descriptions: domain, repos, features, models, services, CLI.
  - Key design decisions: SQLite-first, Protocol-based DI, immutable domain
    objects, feature versioning.
  - Pointer to `fbm.toml` for configuration and `docs/plans/` for roadmaps.
- Keep it under 150 lines — a quick orientation, not exhaustive documentation.

## Success Criteria

- No `assert ... is not None` guards for constructor-injected dependencies.
- `IngestContainer` properties are cached (one instance per container lifetime).
- CI runs on every push/PR and enforces the same checks as pre-commit.
- `docs/architecture.md` exists and covers the data flow end-to-end.
