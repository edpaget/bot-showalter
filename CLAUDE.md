# CLAUDE.md

## Project

Fantasy baseball manager. Python 3.14+, uses `uv` for dependency management. Tests use pytest.

## Development Commands

- **Run tests:** `uv run pytest` (runs in parallel via xdist, randomized order via pytest-randomly)
- **Reproduce test ordering:** `uv run pytest -p randomly -p no:xdist --randomly-seed=SEED` (use the seed printed at the top of a failing run)
- **Lint:** `uv run ruff check src tests`
- **Format:** `uv run ruff format src tests`
- **Type check:** `uv run ty check src tests`
- **Coverage report:** `uv run pytest --cov` (add `--cov-report=html` for HTML output in `htmlcov/`)
- **Fast tests only:** `uv run pytest -m "not slow"` (skips ML model training tests)
- **Install deps:** `uv sync`
- **CI:** GitHub Actions runs the full quality gate (format, lint, type check, tests + coverage) on every push to main and on PRs.

## Code Style

- Use Python type annotations everywhere: function signatures, return types, variable declarations where not obvious. Use `typing` module types and modern syntax (e.g., `str | None` over `Optional[str]`).
- Follow TDD: write failing tests first, then implement the minimum code to pass, then refactor. Tests go in `tests/` mirroring the `src/` structure.
- Favor cohesive modules with low coupling. Use dependency injection to manage dependencies between components — accept collaborators as constructor/function parameters rather than importing and instantiating them directly.
- In tests, prefer injecting fakes/stubs via constructors over `monkeypatch` or `mock.patch`. Since dependencies are typed with Protocols, constructor injection lets the type checker verify test doubles match the real interface. Reserve `monkeypatch` for environment variables and similar global state.
- **All imports must be at the top level of the file.** Never place `import` or `from … import` inside functions, methods, conditionals, or any other nested scope. This is enforced by ruff rule `PLC0415`.
- Lint with ruff using the project's configured rule set.

## Planning

This project uses two tiers of planning:

1. **Roadmaps** (`/roadmap` skill) — High-level multi-phase plans written to `docs/plans/<topic>.md`. These describe what to build and in what order, but not implementation detail.
2. **Phase plans** (built-in plan mode) — When implementing a specific phase from a roadmap, use plan mode to design the detailed implementation steps, then execute after approval.

When asked to "create a plan", "write a plan", or "plan out" a feature — produce a **roadmap** document. Do NOT start implementing or exploring code for implementation purposes unless explicitly asked to implement.

When implementing from a roadmap or plan document, read it first and implement exactly what it specifies. Do not expand scope beyond the plan unless asked. **Every roadmap phase must be implemented in its own worktree** — see [Worktree Workflow](#worktree-workflow) below.

**When asked to implement a roadmap phase**, follow the two-step flow in [Worktree Workflow](#worktree-workflow): first plan (in plan mode), then dispatch a headless agent. Do NOT start implementing code directly.

## Implementation Discipline

When executing a plan (after plan-mode approval):

- Follow TDD: write the failing test first, then the minimum code to pass.
- After each major step, run `uv run pytest` to verify.
- When all steps are complete, commit. Pre-commit hooks run the full quality gate automatically (ruff format, ruff check, ty check, pytest).
- Fix any failures before re-committing.
- Commit with a conventional commit message referencing what was done.
- Coverage is enforced in CI via `fail_under`. Run `uv run pytest --cov` locally to check before pushing.

## Worktree Workflow

**Every roadmap phase must be implemented in its own git worktree.** Never implement a phase directly on `main`. This ensures phases can be developed in parallel without interfering with each other.

### Branch naming

`roadmap/<roadmap-name>/phase-<n>` (e.g., `roadmap/worktree-workflow/phase-1`).

### Two-step implementation flow

This is the primary workflow for implementing roadmap phases:

1. **Plan** — Enter plan mode, read the roadmap doc, explore relevant code, produce a detailed implementation plan, and get user approval.
2. **Dispatch** — Run `./scripts/gwt-implement.sh <roadmap-name> <phase-number>` to create a worktree and launch a headless Claude Code agent that implements the phase in the background. Use `--dry-run` to preview the prompt and command first. Follow progress with `tail -f` on the log file.

### Manual worktree workflow

Fallback for cases where headless dispatch isn't suitable (e.g., interactive exploration, debugging):

- **Create a worktree:** `./scripts/gwt.sh roadmap/<roadmap-name>/phase-<n>`. This sets up an isolated directory with its own `.venv/`, symlinked `data/` and `artifacts/`, and Claude Code settings.
- **Start Claude Code in the worktree:** `claude --cwd ../fbm-roadmap-<roadmap-name>-phase-<n>`.
- **Implement the phase** in the worktree, committing as normal on the phase branch.

### Merging and cleanup

- **Merge back to main** after the phase is complete: from the main worktree, `git merge --ff-only roadmap/<roadmap-name>/phase-<n>` (rebase first if needed to keep history linear).
- **Clean up** the worktree: `./scripts/gwt-remove.sh --delete-branch ../fbm-roadmap-<roadmap-name>-phase-<n>`.
- Phases that depend on earlier phases should branch from main after the dependency has been merged.

## Git Conventions

- Write commits in the **Conventional Commits** style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, etc.
- Keep commit subjects under 72 characters. Use the body for additional detail when needed.
- Keep history linear — no merge commits. Rebase feature branches onto `main` before merging with `git merge --ff-only`.
- Pre-commit hooks enforce the full quality gate (format, lint, type check, tests). Do not skip hooks with `--no-verify`.
- **Always combine `git add` and `git commit` in a single chained command** (e.g., `git add file1 file2 && git commit -m "…"`). Never stage files in a separate step from committing — this avoids conflicts with manually staged files across concurrent agents.
