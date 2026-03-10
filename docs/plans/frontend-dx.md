# Frontend DX Roadmap

The frontend (React 19 + TypeScript + Vite + Apollo Client) has no automated quality gates — no linting, no formatting enforcement, no CI checks, and no pre-commit hooks for TypeScript files. Frontend types are manually maintained duplicates of the Strawberry GraphQL schema, creating drift risk. CLAUDE.md documents only Python tooling, leaving no guidance for frontend development.

This roadmap adds frontend linting/formatting (Biome), GraphQL codegen for automatic type sync, CI integration, conditional pre-commit hooks that trigger only when TypeScript files change, and CLAUDE.md documentation for the full frontend workflow.

## Status

| Phase | Status |
|-------|--------|
| 1 — CLAUDE.md frontend docs | done (2026-03-09) |
| 2 — Biome linting and formatting | done (2026-03-09) |
| 3 — Frontend CI integration | not started |
| 4 — Conditional pre-commit hooks | not started |
| 5 — GraphQL codegen | not started |

## Phase 1: CLAUDE.md frontend docs

Add frontend development commands and conventions to CLAUDE.md so Claude (and developers) know how to build, test, and work with the frontend.

### Context

CLAUDE.md documents Python commands (`uv run pytest`, `ruff check`, etc.) but has zero frontend coverage. There's no mention of `bun`, Vite, Vitest, or any frontend workflow. This means Claude doesn't know to run frontend tests after changes, doesn't know the dev server setup, and can't verify frontend builds.

### Steps

1. Add a "Frontend Development Commands" section to CLAUDE.md covering: `bun install`, `bun run dev`, `bun run build`, `bun test`, `bun run test:watch`.
2. Document the Vite proxy setup (dev server proxies `/graphql` to `http://127.0.0.1:8000`).
3. Add frontend code style conventions: component file naming, test file co-location, Apollo mocking patterns, TailwindCSS usage.
4. Note that `bun` (not npm/yarn) is the package manager for the frontend.

### Acceptance criteria

- CLAUDE.md has a frontend section with all dev commands.
- Running each documented command from `frontend/` works as described.
- Frontend conventions (naming, testing, styling) are documented.

## Phase 2: Biome linting and formatting

Add Biome as the frontend linter and formatter, replacing the current zero-enforcement state.

### Context

The frontend has no ESLint, Prettier, or Biome configuration. There's no automated way to catch unused imports, formatting inconsistencies, or common React mistakes. Biome is chosen over ESLint+Prettier because it's a single tool, fast, and has good TypeScript/React support out of the box.

### Steps

1. Install Biome as a dev dependency: `bun add -d @biomejs/biome`.
2. Create `frontend/biome.json` with rules matching the project's strictness (strict mode, no unused imports/variables — aligning with the existing `tsconfig.json` settings).
3. Add `lint` and `format` scripts to `package.json`: `"lint": "biome check src"`, `"format": "biome format --write src"`, `"lint:fix": "biome check --write src"`.
4. Run Biome on the existing codebase and fix any violations.
5. Update CLAUDE.md frontend section to reference the new lint/format commands.

### Acceptance criteria

- `bun run lint` runs Biome and reports violations (or passes clean).
- `bun run format` formats all frontend source files.
- All existing frontend code passes `bun run lint` without errors.
- CLAUDE.md documents the lint and format commands.

## Phase 3: Frontend CI integration

Add frontend quality checks to the GitHub Actions CI pipeline.

### Context

CI currently runs only Python checks (ruff, ty, pytest with coverage). Frontend code can break without any CI signal — TypeScript compile errors, test failures, and lint violations all go undetected until someone manually runs them.

### Steps

1. Add a `frontend` job to `.github/workflows/ci.yml` that runs in parallel with the existing Python job.
2. The job should: checkout code, install bun, run `bun install --frozen-lockfile`, then run `bun run lint`, `tsc -b` (type check), and `bun test`.
3. Ensure the job only triggers when frontend files change (use `paths` filter or run unconditionally if simpler).
4. Add the frontend job to any branch protection rules if applicable.

### Acceptance criteria

- Push to main with a frontend TypeScript error fails CI.
- Push to main with a frontend test failure fails CI.
- Push to main with a Biome lint violation fails CI.
- Frontend CI job runs in parallel with the Python job (not sequentially).

## Phase 4: Conditional pre-commit hooks

Add pre-commit hooks that run frontend checks only when TypeScript/frontend files are staged.

### Context

The existing pre-commit config runs Python checks (ruff format, ruff check, ty, pytest) on every commit. Frontend developers (or Claude) changing `.ts`/`.tsx` files get no pre-commit feedback. Adding unconditional frontend hooks would slow down Python-only commits. The hooks should be conditional — only run when frontend files are staged.

### Steps

1. Add a local hook to `.pre-commit-config.yaml` that runs Biome check on staged frontend files. Use `types_or: [ts, tsx]` and `files: ^frontend/` to scope it.
2. Add a local hook that runs `tsc -b` in the `frontend/` directory, triggered when any `.ts` or `.tsx` file under `frontend/` is staged. Since `tsc` checks all files (not just staged ones), use `pass_filenames: false` and gate on `files: ^frontend/`.
3. Add a local hook that runs `bun test` in the `frontend/` directory, also gated on `files: ^frontend/` with `pass_filenames: false`.
4. Test that committing only Python files skips frontend hooks.
5. Test that committing a `.tsx` file triggers all three frontend hooks.
6. Update CLAUDE.md to document the pre-commit behavior.

### Acceptance criteria

- Committing only `.py` files does not run any frontend hooks.
- Committing a `frontend/src/*.tsx` file runs Biome, tsc, and vitest.
- A Biome violation in a staged `.tsx` file blocks the commit.
- A TypeScript compile error blocks the commit.
- A failing frontend test blocks the commit.

## Phase 5: GraphQL codegen

Replace hand-written frontend TypeScript types with auto-generated types from the Strawberry GraphQL schema.

### Context

The frontend has ~200 lines of manually maintained TypeScript interfaces in `frontend/src/types/` that mirror the Strawberry types in `src/fantasy_baseball_manager/web/types.py`. When the backend schema changes (new fields, renamed types, new queries), the frontend types must be updated by hand. This is error-prone — mismatches cause runtime errors that TypeScript can't catch because the types are lies.

### Steps

1. Add a Strawberry schema export command: a small script or CLI command that runs `strawberry export-schema fantasy_baseball_manager.web.schema:schema > frontend/schema.graphql`. Alternatively, add a `bun run schema:export` script that calls the Python backend.
2. Install `@graphql-codegen/cli`, `@graphql-codegen/typescript`, `@graphql-codegen/typescript-operations`, and `@graphql-codegen/typed-document-node` as frontend dev dependencies.
3. Create `frontend/codegen.ts` configuration that reads `frontend/schema.graphql` and the `frontend/src/graphql/*.ts` query/mutation/subscription documents, and outputs generated types to `frontend/src/generated/graphql.ts`.
4. Add `"codegen": "graphql-codegen"` script to `package.json`.
5. Run codegen and replace manual type imports in components with generated types. Remove the hand-written `frontend/src/types/` files (or keep only non-GraphQL types like utility types).
6. Update Apollo Client hooks to use typed document nodes (provides end-to-end type safety from query to component).
7. Add codegen output to `.gitignore` or commit it (team preference — committing is simpler for this project).
8. Add a CI step that runs codegen and checks that the output matches what's committed (prevents drift).
9. Update CLAUDE.md with the codegen workflow: when to run it, how schema changes propagate.

### Acceptance criteria

- `bun run codegen` generates TypeScript types from the Strawberry schema.
- All frontend components use generated types — no hand-written GraphQL type interfaces remain.
- Apollo Client hooks (`useQuery`, `useMutation`, `useSubscription`) use typed document nodes.
- Adding a field to a Strawberry type and running codegen produces the updated TypeScript type without manual edits.
- CI fails if codegen output is stale (generated file doesn't match committed version).

## Ordering

Phases are ordered by dependency and effort:

1. **Phase 1** (CLAUDE.md docs) has no dependencies and unblocks all other phases by documenting existing commands. Do this first.
2. **Phase 2** (Biome) has no dependencies but is a prerequisite for phases 3 and 4 (which run Biome in CI and pre-commit).
3. **Phase 3** (CI) depends on phase 2 for the lint step. Could run in parallel with phase 4.
4. **Phase 4** (pre-commit hooks) depends on phase 2. Can run in parallel with phase 3.
5. **Phase 5** (GraphQL codegen) is independent of phases 2-4 but is the most complex. Best done last when the quality infrastructure is in place to catch issues.
