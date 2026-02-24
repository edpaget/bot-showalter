# Architectural Principles

This document codifies the architectural principles that govern the fantasy baseball manager codebase. These principles are intended to be enforced — by convention, by tests, by linting, and by automated review.

---

## 1. Module Decoupling via Protocols and Dependency Injection

Modules must not directly depend on each other's concrete implementations. Instead:

- **Declare dependencies as protocols.** When a module needs a collaborator, it depends on a `Protocol` (defined locally or in a shared protocols module), never on a concrete class from another package.
- **Wire at the boundary.** The outer ports of the system — CLI commands (`cli/`), the LLM agent (`agent/` + `tools/`), HTTP servers — are composition roots. They import concrete implementations and wire them together via DI containers or factory functions.
- **Use constructor injection.** Services and other collaborators accept their dependencies as constructor parameters. This makes dependencies explicit, testable, and verifiable by the type checker.

### What this means in practice

- A service in `services/` may depend on `PlayerRepo` (a protocol), but must never import `SqlitePlayerRepo` (a concrete repo).
- `cli/factory.py` and `analysis_container.py` are composition roots — they are *allowed* to import concrete classes from multiple modules and wire them together.
- Domain models in `domain/` must have zero dependencies on infrastructure (repos, HTTP clients, databases).

### Why

Loose coupling lets us swap implementations (e.g., test fakes for real repos), evolve modules independently, and reason about each module in isolation.

---

## 2. All Code Must Be Tested

Every behavior must be covered by automated tests. There are no exceptions for "glue code" or "simple wrappers."

- **Follow TDD.** Write a failing test first, then the minimum code to make it pass, then refactor.
- **Use real infrastructure where cheap.** SQLite is fast and deterministic — use in-memory SQLite for repository tests rather than mocking the database. This tests actual SQL and schema interactions.
- **Prefer constructor-injected fakes over mocks.** Since dependencies are typed with protocols, test doubles constructed via DI let the type checker verify they conform to the real interface. Reserve `monkeypatch` for global state (environment variables, module-level constants).
- **Maintain coverage.** The project enforces a coverage floor (currently 88%) in CI. New code must not drop coverage below this threshold.

### Why

Tests are the primary defense against regressions. Testing against real SQLite (rather than mocks) catches schema mismatches and SQL bugs that mocks would hide. Constructor injection keeps test setup explicit and type-safe.

---

## 3. Secrets via Environment Variables

All secrets — API keys, OAuth tokens, database credentials — must be configurable via environment variables. Never hard-code secrets or commit them to the repository.

- **Environment variables are the canonical source.** Configuration code should read secrets from `os.environ` (or a settings loader that does so).
- **Configuration files may reference defaults** for non-secret settings, but must never contain secret values.
- **`.env` files are gitignored.** Developers may use `.env` files locally, but they must never be committed.

### Why

Environment-variable-based secrets are the standard for twelve-factor apps. They work across local development, CI, and production without code changes, and eliminate the risk of accidentally committing credentials.

---

## 4. Model Generality

No projection model receives special treatment. Any capability implemented for one model must be generalizable to all models that share the same protocol surface.

- **Models conform to protocols.** The `Model`, `Preparable`, `Trainable`, and `Evaluable` protocols define the contract. A model opts into capabilities by implementing the relevant protocol — it does not get bespoke infrastructure.
- **The registry is the only discovery mechanism.** CLI commands and services interact with models through the registry (`models/registry.py`), never by importing a specific model class directly (except in the model's own package).
- **Shared utilities live in shared modules.** If a training helper, feature transform, or evaluation metric is useful to more than one model, it belongs in a shared module (e.g., `models/gbm_training.py`, `features/`), not buried inside a single model's package.
- **No model-specific CLI commands.** CLI commands operate on model *names* passed as arguments, dispatching through the registry. There must be no command that hard-codes a specific model.

### Why

The system's value comes from comparing and composing multiple projection approaches. If a model gets special-case code, that code becomes a maintenance burden and a barrier to adding new models. Generality keeps the model ecosystem open for extension.

---

## 5. Interaction Mode Generality

No interaction mode (CLI, LLM agent, HTTP server) receives privileged access to business logic. Any capability exposed through one mode must be available — or trivially exposable — through any other.

- **Business logic lives in services.** Services are mode-agnostic. They accept domain objects and return domain objects. They never import from `cli/`, `agent/`, `tools/`, or HTTP-layer code.
- **Interaction modes are thin adapters.** CLI commands, agent tools, and HTTP handlers parse input, call services, and format output. They contain no business logic of their own.
- **Shared formatting is acceptable** when modes need similar output (e.g., table rendering), but formatting code must not be coupled to a single mode.

### Why

Users interact with the system through different interfaces depending on context — the CLI for batch operations, the agent for conversational exploration, the live server during drafts. If business logic is trapped inside one mode's code, the other modes can't use it, leading to duplication or feature gaps.

---

## 6. Module Public API via Re-exports

Each package should present a clear, consolidated public API through its `__init__.py`. Consumers should be able to import from the package directly, without reaching into submodules.

- **Re-export public symbols from `__init__.py`.** If `repos/` exposes `PlayerRepo`, `ProjectionRepo`, and `BattingStatsRepo`, a consumer should be able to write `from fantasy_baseball_manager.repos import PlayerRepo`.
- **Internal submodules are implementation details.** Consumers must not need to know that `PlayerRepo` lives in `repos/player_repo.py` — the package boundary is the contract.
- **Keep re-exports intentional.** Only export symbols that are part of the package's public API. Use `__all__` where helpful to make the boundary explicit.

### Why

Consolidated re-exports make packages easier to use, reduce import churn when internals are refactored, and make dependency relationships between packages visible at a glance. When every import goes through the package root, it's straightforward to audit and enforce coupling rules.

---

## 7. Domain Models Are Pure Data

Domain models are immutable value objects with no attached behavior. They carry data between layers but do not act on it.

- **Use `@dataclass(frozen=True)`.** All domain models are frozen dataclasses. Immutability prevents accidental mutation as objects pass through services, repos, and interaction layers.
- **No methods beyond what dataclass provides.** Domain classes define fields, not behavior. Logic that operates on domain objects belongs in services or standalone functions — not in methods on the model itself.
- **No infrastructure dependencies.** Domain modules may only import from the standard library (`dataclasses`, `typing`, `enum`, `statistics`, etc.) and from other domain modules. They must never import from repos, services, ingest, or any infrastructure layer.

### Why

Pure data objects are trivial to construct in tests, safe to pass across layer boundaries, and free of hidden side effects. Keeping behavior in services (rather than on models) avoids the "god object" pattern where domain classes accumulate methods from every use case that touches them.

---

## 8. Unidirectional Layer Dependencies

Dependencies flow strictly downward through the layer stack. Higher layers may import lower layers; lower layers must never import higher layers.

```
CLI / Agent / HTTP   (interaction layer — composition roots)
        ↓
    Services          (business logic)
        ↓
      Repos           (data access)
        ↓
     Domain           (pure data)
```

- **Domain imports nothing above it.** Domain models depend only on the standard library and other domain modules.
- **Repos import domain, never services.** Repos convert between domain objects and storage. They have no knowledge of business rules.
- **Services import repo protocols and domain, never interaction-layer code.** A service must never import from `cli/`, `agent/`, `tools/`, or HTTP code.
- **Interaction layers import everything below.** CLI commands, agent tools, and HTTP handlers are the composition roots — they wire services and repos together.

### Why

Unidirectional flow prevents circular dependencies and keeps each layer independently testable. If a repo depends on a service, or a service depends on a CLI module, the dependency graph becomes tangled and changes ripple unpredictably. Strict layering ensures that changes to the CLI never break domain logic, and changes to domain models never require changes to infrastructure above.

---

## 9. Raw Parameterized SQL

Repos use hand-written SQL with parameterized queries. There is no ORM and no query builder.

- **Always use `?` placeholders.** Values are passed as parameters to `conn.execute()`, never interpolated into SQL strings. This prevents SQL injection and ensures correct type handling.
- **Use `ON CONFLICT ... DO UPDATE` for upserts.** The standard write pattern is insert-or-update in a single statement, not check-then-insert.
- **Convert rows to domain objects explicitly.** Each repo provides a `_row_to_<entity>()` helper that maps database rows to frozen dataclasses. The domain layer never sees raw rows.
- **Build dynamic SQL from column names, not values.** When queries need variable column lists (e.g., stat columns), build the SQL by joining column name constants — never by interpolating user-supplied values.

### Why

Raw SQL keeps the data layer transparent and debuggable. Every query is visible in the repo file, with no hidden ORM magic generating unexpected statements. Parameterized queries prevent injection. Explicit row-to-domain mapping keeps the boundary between storage representation and domain representation clear.

---

## 10. Result Types for Recoverable Errors

Operations that can fail in expected, recoverable ways return `Result[T, E]` rather than raising exceptions. Exceptions are reserved for programming errors and unrecoverable failures.

- **`Result[T, E]` for expected failures.** When an operation has a well-defined failure mode (e.g., ingestion of an external source that may be unavailable), it returns `Ok[T]` on success or `Err[E]` on failure. The caller is forced to handle both cases.
- **Exceptions for bugs and infrastructure failures.** Unexpected conditions (violated invariants, database corruption, programming errors) raise exceptions from the `FbmException` hierarchy. These propagate up to the interaction layer for reporting.
- **Semantic exceptions at repo boundaries.** When a database constraint violation represents a domain concept (e.g., duplicate player), the repo catches the low-level error and raises a domain-specific exception (e.g., `PlayerConflictError`).

### Why

Result types make failure handling explicit and compiler-checkable. The caller cannot accidentally ignore an error the way it can with an uncaught exception. Reserving exceptions for truly exceptional conditions keeps the happy path clean and the error-handling paths visible.

---

## 11. Favor Simple Functional Interfaces

When defining a dependency boundary, prefer the simplest interface that works: a `Callable`, a single-method protocol, or a protocol using `__call__`. Reach for multi-method protocols only when the methods are genuinely cohesive and always used together.

- **One operation → one interface.** If a collaborator only needs to do one thing (transform a row, train a model, evaluate accuracy), define it as a `Callable` type alias, a `__call__` protocol, or a single-method protocol. Do not bundle unrelated operations into one interface just because they happen to live on the same object.
- **Compose small protocols rather than inflating large ones.** When a class supports multiple capabilities, define each as its own single-method protocol (`Preparable`, `Trainable`, `Evaluable`) and let the class implement several. Callers depend only on the slice they need.
- **Repositories are an acceptable exception.** Repo protocols naturally group related query methods around a single entity (upsert, get-by-id, get-by-season, etc.). This cohesion is inherent to the data-access pattern. Even so, watch for protocols that grow beyond their entity boundary — if a repo protocol mixes concerns (e.g., projections and distributions), consider splitting it.
- **Prefer `Callable` for stateless transforms.** When a dependency is a pure function with no setup or teardown, a `Callable` type alias is clearer than a single-method protocol and composes naturally with lambdas and closures.

### Why

Small interfaces are easier to implement, easier to fake in tests, and harder to misuse. A function that accepts `Callable[[Row], Row]` can be tested with a lambda; one that accepts a five-method protocol requires a full test double. Single-method protocols also make dependency graphs legible — you can see at a glance exactly what capability each collaborator provides.
