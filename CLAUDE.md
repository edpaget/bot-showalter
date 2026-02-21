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

This project uses two levels of planning:

1. **Roadmaps** (`/roadmap` skill) — High-level multi-phase plans written to `docs/plans/<topic>.md`. These describe what to build and in what order, but not implementation detail. Each roadmap has a **Status** table that tracks phase progress. See [`docs/plans/INDEX.md`](docs/plans/INDEX.md) for an overview of all roadmaps and their dependencies.
2. **Plan mode** (built-in) — Used when starting implementation of a roadmap phase. Enter plan mode, explore the code, produce a detailed implementation plan, and get user approval before writing code. The plan lives only in the plan-mode session — do not write separate phase-plan files to disk.

When asked to "create a plan", "write a plan", or "plan out" a feature — produce a **roadmap** document. Do NOT start implementing or exploring code for implementation purposes unless explicitly asked to implement.

When implementing from a roadmap, use the `/implement` skill (e.g., `/implement tier-generator phase 1`). It handles reading the roadmap, entering plan mode, worktree setup, and status updates. The full sequence is:

1. Read the roadmap and locate the target phase's steps and acceptance criteria.
2. Enter plan mode to explore the code and design the implementation approach.
3. After plan-mode approval, implement in a worktree (see Worktree Workflow).
4. Do not expand scope beyond the roadmap phase unless asked.
5. After the phase lands, update the roadmap's Status table (mark the phase `done (<date>)`) and update `docs/plans/INDEX.md` if progress changes affect the dependency graph or status summary.

## Implementation Discipline

When executing a plan (after plan-mode approval):

- Follow TDD: write the failing test first, then the minimum code to pass.
- After each major step, run `uv run pytest` to verify.
- When all steps are complete, commit. Pre-commit hooks run the full quality gate automatically (ruff format, ruff check, ty check, pytest).
- Fix any failures before re-committing.
- Commit with a conventional commit message referencing what was done.
- Coverage is enforced in CI via `fail_under`. Run `uv run pytest --cov` locally to check before pushing.
- **Before the final commit, verify each acceptance criterion from the roadmap phase.** Walk the list explicitly — confirm each one is satisfied by the implementation or tests. If a criterion isn't met, finish it before committing.

## Worktree Workflow

Implement each roadmap phase in its own worktree to avoid working directly on `main`. Use the built-in `EnterWorktree` tool to create an isolated worktree and switch into it. After the phase is complete, merge back to main with `git merge --ff-only` (rebase first if needed to keep history linear). Then clean up the worktree branch:

```bash
git worktree remove .claude/worktrees/<name>
git branch -d <branch-name>
```

## Git Conventions

- Write commits in the **Conventional Commits** style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, etc.
- Keep commit subjects under 72 characters. Use the body for additional detail when needed.
- Keep history linear — no merge commits. Rebase feature branches onto `main` before merging with `git merge --ff-only`.
- Pre-commit hooks enforce the full quality gate (format, lint, type check, tests). Do not skip hooks with `--no-verify`.
- **Always combine `git add` and `git commit` in a single chained command** (e.g., `git add file1 file2 && git commit -m "…"`). Never stage files in a separate step from committing — this avoids conflicts with manually staged files across concurrent agents.
