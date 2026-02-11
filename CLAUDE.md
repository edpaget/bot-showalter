# CLAUDE.md

## Project

Fantasy baseball manager. Python 3.13+, uses `uv` for dependency management. Tests use pytest (~2300 tests); the full suite runs in ~30 seconds.

## Development Commands

- **Run tests:** `uv run pytest`
- **Lint:** `uv run ruff check src tests`
- **Format:** `uv run black src tests`
- **Type check:** `uv run ty check src tests`
- **Install deps:** `uv sync`

## Code Style

- Use Python type annotations everywhere: function signatures, return types, variable declarations where not obvious. Use `typing` module types and modern syntax (e.g., `str | None` over `Optional[str]`).
- Follow TDD: write failing tests first, then implement the minimum code to pass, then refactor. Tests go in `tests/` mirroring the `src/` structure.
- Favor cohesive modules with low coupling. Use dependency injection to manage dependencies between components — accept collaborators as constructor/function parameters rather than importing and instantiating them directly.
- In tests, prefer injecting fakes/stubs via constructors over `monkeypatch` or `mock.patch`. Since dependencies are typed with Protocols, constructor injection lets the type checker verify test doubles match the real interface. Reserve `monkeypatch` for environment variables and similar global state.
- Lint with ruff using the project's configured rule set.

## Planning

This project uses two tiers of planning:

1. **Roadmaps** (`/roadmap` skill) — High-level multi-phase plans written to `docs/plans/<topic>.md`. These describe what to build and in what order, but not implementation detail.
2. **Phase plans** (built-in plan mode) — When implementing a specific phase from a roadmap, use plan mode to design the detailed implementation steps, then execute after approval.

When asked to "create a plan", "write a plan", or "plan out" a feature — produce a **roadmap** document. Do NOT start implementing or exploring code for implementation purposes unless explicitly asked to implement.

When implementing from a roadmap or plan document, read it first and implement exactly what it specifies. Do not expand scope beyond the plan unless asked.

## Implementation Discipline

When executing a plan (after plan-mode approval):

- Follow TDD: write the failing test first, then the minimum code to pass.
- After each major step, run `uv run pytest` to verify.
- When all steps are complete, run the full quality gate: `uv run pytest`, `uv run ruff check src tests`, `uv run ty check src tests`.
- Fix any failures before committing.
- Commit with a conventional commit message referencing what was done.

## Git Conventions

- Write commits in the **Conventional Commits** style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, etc.
- Keep commit subjects under 72 characters. Use the body for additional detail when needed.
- Keep history linear — no merge commits. Rebase feature branches onto `main` before merging with `git merge --ff-only`.
- **Always run `uv run ty check src tests` before committing** and fix any type errors. Do not commit code that introduces new type-check failures.
