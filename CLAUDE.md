# CLAUDE.md

## Project

Fantasy baseball manager. Python 3.13+, uses `uv` for dependency management.

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
- Lint with ruff using the project's configured rule set.

## Git Conventions

- Write commits in the **Conventional Commits** style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, etc.
- Keep commit subjects under 72 characters. Use the body for additional detail when needed.
