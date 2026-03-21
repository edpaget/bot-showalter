# CLAUDE.md

## Project

Fantasy baseball manager. Python 3.14+, uses `uv` for dependency management. Tests use pytest.

## Development Commands

- **Run tests:** `uv run pytest` (runs in parallel via xdist, randomized order via pytest-randomly)
- **Reproduce test ordering:** `uv run pytest -p randomly -p no:xdist --randomly-seed=SEED` (use the seed printed at the top of a failing run)
- **Lint:** `uv run ruff check src tests`
- **Format:** `uv run ruff format src tests`
- **Type check:** `uv run ty check src tests`
- **Coverage report:** `uv run pytest --cov` (add `--cov-report=html` for HTML output in `htmlcov/`). Always include slow tests when checking coverage — do **not** combine `--cov` with `-m "not slow"`, as that lowers the total and fails the `fail_under` threshold.
- **Fast tests only:** `uv run pytest -m "not slow"` (skips ML model training tests)
- **Install deps:** `uv sync`
- **CI:** GitHub Actions runs the full quality gate (format, lint, type check, tests + coverage) on every push to main and on PRs.

## Frontend Development

### Commands

All commands run from `frontend/`.

- **Install deps:** `bun install` — lock file is `bun.lock`.
- **Dev server:** `bun run dev` (Vite dev server; proxies `/graphql` to `http://127.0.0.1:8000` including WebSocket for subscriptions)
- **Build:** `bun run build` (TypeScript check + Vite production build)
- **Run tests:** `bun run test` (Vitest, jsdom environment). **Not** `bun test` — that invokes bun's native runner which skips the Vitest/jsdom config.
- **Watch tests:** `bun run test:watch`
- **Lint:** `bun run lint` (Biome check — linting + format verification)
- **Format:** `bun run format` (auto-format all frontend source files)
- **Lint + fix:** `bun run lint:fix` (auto-fix lint and format issues)
- **Package manager:** `bun` (not npm/yarn).
- **Codegen:** `bun run codegen` — regenerates `frontend/src/generated/graphql.ts` from `frontend/schema.graphql` + `frontend/src/graphql/documents/*.ts`. Run after changing any GraphQL query/mutation/subscription or after updating the backend schema.
- **Schema export:** `bun run schema:export` (or `uv run python scripts/export_schema.py > frontend/schema.graphql` from repo root) — exports the Strawberry GraphQL schema as SDL. Run when backend schema types change.

### GraphQL Codegen

Types are auto-generated from the Strawberry schema. **Do not hand-write GraphQL type interfaces.**

- **Schema source:** `frontend/schema.graphql` (committed, exported from Python backend)
- **Generated file:** `frontend/src/generated/graphql.ts` (committed, excluded from Biome lint)
- **Document definitions:** `frontend/src/graphql/documents/` — raw `gql` tag definitions that codegen reads
- **Re-export barrels:** `frontend/src/graphql/queries.ts`, `mutations.ts`, `subscriptions.ts` — re-export typed document nodes from `../generated/graphql` with familiar names (e.g., `BOARD_QUERY`)
- **Position utility:** `frontend/src/lib/position.ts` — runtime `Position` const + `displayPosition()` function
- **Workflow:** When adding/changing a query: edit the document in `src/graphql/documents/`, run `bun run codegen`, then import the typed document node from the barrel re-export
- **CI:** The frontend CI job checks codegen freshness — it exports the schema, runs codegen, and fails if `src/generated/` has uncommitted changes

### Pre-commit Hooks

Committing `.ts`/`.tsx` files under `frontend/` automatically triggers three frontend hooks: Biome check, tsc type check, and Vitest. Python-only commits skip these entirely. This is controlled by `types_or: [ts, tsx]` and `files: ^frontend/` filters in `.pre-commit-config.yaml`.

### Code Style

- **Component files:** PascalCase (`PlayerDrawer.tsx`). Named exports, not default exports.
- **Test co-location:** Tests live next to components (`PlayerDrawer.test.tsx`).
- **Styling:** TailwindCSS v4 utility classes directly in `className`. No CSS modules or styled-components. Tailwind is configured via the `@tailwindcss/vite` plugin — no separate `tailwind.config` or `postcss.config` files.
- **State management:** Apollo Client cache + React Context. No Redux or external state library.
- **Apollo mocking in tests:** Use `MockedProvider` from `@apollo/client/testing` with `addTypename: false`. Define mock responses as `MockedResponse` objects with `request` (query + variables) and `result` (data). Typed document nodes use `Exact<>` variable types — include all variable fields (pass `null` for optional ones) to match.
- **GraphQL types:** All TypeScript types for GraphQL are auto-generated in `src/generated/graphql.ts`. Import types (e.g., `BoardQuery`, `DraftBoardRowType`) from `../generated/graphql`. Apollo hooks infer types from typed document nodes — no manual generic parameters needed.
- **GraphQL queries/mutations/subscriptions:** Re-exported from `frontend/src/graphql/` barrels (`queries.ts`, `mutations.ts`, `subscriptions.ts`). Raw gql definitions live in `frontend/src/graphql/documents/`.

## Code Style

- Use Python type annotations everywhere: function signatures, return types, variable declarations where not obvious. Use `typing` module types and modern syntax (e.g., `str | None` over `Optional[str]`).
- Follow TDD: write failing tests first, then implement the minimum code to pass, then refactor. Tests go in `tests/` mirroring the `src/` structure.
- Favor cohesive modules with low coupling. Use dependency injection to manage dependencies between components — accept collaborators as constructor/function parameters rather than importing and instantiating them directly.
- In tests, prefer injecting fakes/stubs via constructors over `monkeypatch` or `mock.patch`. Since dependencies are typed with Protocols, constructor injection lets the type checker verify test doubles match the real interface. Reserve `monkeypatch` for environment variables and similar global state.
- **All imports must be at the top level of the file.** Never place `import` or `from … import` inside functions, methods, conditionals, or any other nested scope. This is enforced by ruff rule `PLC0415`.
- Lint with ruff using the project's configured rule set.

## Planning

This project uses two levels of planning:

1. **Roadmaps** (`/roadmap` skill) — High-level multi-phase plans managed via the rdm MCP server (`project: fbm`). These describe what to build and in what order, but not implementation detail. Use `mcp__rdm__rdm_roadmap_list` to see all roadmaps and their progress.
2. **Plan mode** (built-in) — Used when starting implementation of a roadmap phase. Enter plan mode, explore the code, produce a detailed implementation plan, and get user approval before writing code. The plan lives only in the plan-mode session — do not write separate phase-plan files to disk.

When asked to "create a plan", "write a plan", or "plan out" a feature — produce a **roadmap** via the rdm MCP tools. Do NOT start implementing or exploring code for implementation purposes unless explicitly asked to implement.

When implementing from a roadmap, use the `/implement` skill (e.g., `/implement tier-generator phase 1`). It handles reading the roadmap, entering plan mode, worktree setup, and status updates. The full sequence is:

1. Read the roadmap phase via `mcp__rdm__rdm_phase_show` and locate the steps and acceptance criteria.
2. Enter plan mode to explore the code and design the implementation approach.
3. After plan-mode approval, implement in a worktree (see Worktree Workflow).
4. Do not expand scope beyond the roadmap phase unless asked.
5. After the phase lands, update phase status via `mcp__rdm__rdm_phase_update` and merge back to main (see Worktree Workflow).

## Layer Dependencies — Models Must Not Import Services

The architecture enforces strict layer boundaries (see `docs/principles.md` §8). The most common mistake is a **model** importing a **service**. This is forbidden and caught by `tests/architecture/test_layer_dependencies.py`.

**When a model needs functionality that lives in a service, follow this recipe:**

1. **Define a Protocol** (or `Callable` type alias) describing the capability the model needs. Put it in the model's own package or in `domain/model_protocol.py` if it's shared. Keep it minimal — one method or `__call__`.
2. **Accept the Protocol as a constructor parameter** on the model class.
3. **Wire it in the composition root** (`cli/factory.py` or `analysis_container.py`). The composition root is the only place that imports both the concrete service and the model, and passes the service as the protocol-typed dependency.

```python
# In models/my_model/model.py — define what you need as a protocol
class ScoreCalculator(Protocol):
    def __call__(self, player_id: int, season: int) -> float: ...

class MyModel:
    def __init__(self, score_calculator: ScoreCalculator) -> None:
        self._score_calculator = score_calculator

# In cli/factory.py — wire the concrete service to the model
from fantasy_baseball_manager.services.scoring import ScoringService
model = MyModel(score_calculator=ScoringService(...))
```

**Never** add exclusions to `KNOWN_EXCEPTIONS` in the arch tests to work around this — the fix is always to introduce a Protocol and inject via the constructor.

The same pattern applies to any layer boundary: services must not import CLI code, repos must not import services, etc. When in doubt, check the `FORBIDDEN_IMPORTS` dict in `tests/architecture/test_layer_dependencies.py`.

## Implementation Discipline

When executing a plan (after plan-mode approval):

- **Before writing data-layer code**, verify the actual shapes you'll work with. Run `.schema <table>` in SQLite to check real column names; make a minimal test request to any external API and inspect the response structure. Do not assume schemas or parameter names from memory — check first.
- Follow TDD: write the failing test first, then the minimum code to pass.
- After each major step, run `uv run pytest` to verify.
- When all steps are complete, commit. Pre-commit hooks run the full quality gate automatically (ruff format, ruff check, ty check, pytest).
- Fix any failures before re-committing.
- Commit with a conventional commit message referencing what was done.
- Coverage is enforced in CI via `fail_under`. Run `uv run pytest --cov` locally to check before pushing.
- **Before the final commit, verify each acceptance criterion from the roadmap phase.** Walk the list explicitly — confirm each one is satisfied by the implementation or tests. If a criterion isn't met, finish it before committing.
- **After any model training or tuning change**, run the before/after comparison protocol from the fbm skill: `compare old new --season YEAR --top 300 --check` and `compare old new --season YEAR` (full population) on at least two holdout seasons. Do not declare improvement unless `--check` passes on all tested seasons for both top-300 and full population.
- **After any valuation formula change**, run `fbm valuations compare old/version new/version --season YEAR --league LEAGUE --check` on at least two holdout seasons. The `--check` flag gates on independent targets (WAR ρ, hit rates), not circular ZAR$ metrics.

## Worktree Workflow

Use one worktree per roadmap, not per phase. This avoids losing the session's working directory — `EnterWorktree` switches the cwd into the worktree, and there is no way to switch back, so removing the worktree mid-session breaks things.

When working in a worktree, all file reads, searches, and code exploration must use **relative paths or the worktree's absolute path** (check with `pwd`). Never hardcode `/Users/edward/Projects/fbm` or `~/.claude/worktrees/…` paths — these will be wrong. The worktree is your cwd after `EnterWorktree`; use relative paths like `src/`, `tests/`, `docs/` for all tool calls (Read, Glob, Grep, etc.). The only reason to `cd` to the main repo is to run `git merge --ff-only` when landing a phase.

**Starting a roadmap:** Use `EnterWorktree` with the roadmap name (e.g., `name: "player-eligibility"`). This creates the worktree and switches the session into it. Do not update the roadmap status to mark a phase as in-progress until after entering the worktree — plan tracking changes should be committed on the worktree branch, not on main.

**After each phase lands:**

1. **Update plan tracking.** Mark the phase done via `mcp__rdm__rdm_phase_update` with `status: "done"`.
2. **Merge back to main from the main repo checkout:**

```bash
git rebase main                 # rebase onto main to ensure fast-forward is possible
cd /Users/edward/Projects/fbm  # switch to the main repo where main is checked out
git merge --ff-only <branch>    # fast-forward main to the worktree branch tip
cd -                            # return to the worktree
git checkout -B <roadmap> main  # create a fresh branch for the next phase, based on main's tip
```

3. Continue with the next phase in the same worktree, or end the session. The worktree stays alive — the user will be prompted to keep or remove it when the session exits.

## Git Conventions

- Write commits in the **Conventional Commits** style: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, etc.
- Keep commit subjects under 72 characters. Use the body for additional detail when needed.
- Keep history linear — no merge commits. Rebase feature branches onto `main` before merging with `git merge --ff-only`.
- Pre-commit hooks enforce the full quality gate (format, lint, type check, tests). Do not skip hooks with `--no-verify`.
- **Always combine `git add` and `git commit` in a single chained command** (e.g., `git add file1 file2 && git commit -m "…"`). Never stage files in a separate step from committing — this avoids conflicts with manually staged files across concurrent agents.
- **Pre-commit checklist** (follow every time):
  1. Run `uv run ruff format src tests` before staging so pre-commit hooks don't reformat and force a re-stage cycle.
  2. `git add` only the files related to the current task — name them explicitly, never use `git add .` or `git add -A`.
  3. Run `git diff --cached --name-only` and verify every listed file belongs to this change. Remove anything unrelated with `git reset HEAD <file>`.
  4. If pre-commit hooks still modify files (e.g., ruff finds a new issue), re-stage the changed files and create a **new** commit — do not amend.
