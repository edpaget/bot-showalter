# Connection Pool Repos Roadmap

Refactor all SQLite-backed repos to accept a `ConnectionPool` instead of a raw `sqlite3.Connection`. Currently every repo stores a single connection in `self._conn`, which breaks when LangGraph dispatches agent tool calls across threads. The connection pool (`db/pool.py`) already exists but is unused. This roadmap wires it through the entire repo layer for thread-safe database access.

The key design challenge is that repos serve two modes: **read-only** (agent tools, queries) and **write** (ingest, CLI commands with explicit `conn.commit()` calls). The pool's `connection()` context manager rolls back on exit, which is correct for reads but breaks the current write pattern where the caller commits across multiple repo calls. To handle both, we introduce a `ConnectionProvider` protocol with two implementations: one backed by the pool (per-call checkout) and one backed by a scoped connection (for write transactions).

## Status

| Phase | Status |
|-------|--------|
| 1 — ConnectionProvider protocol | done (2026-03-05) |
| 2 — Migrate repos | not started |
| 3 — Migrate composition roots and tests | not started |
| 4 — Wire pool into agent path | not started |

## Phase 1: ConnectionProvider protocol

Define the abstraction that repos will depend on, plus two concrete implementations.

### Context

Repos currently accept `sqlite3.Connection` directly. We need an abstraction that supports both pool-backed (thread-safe, per-call checkout) and single-connection (write transactions with caller-managed commit/rollback) usage. The `ConnectionPool` class in `db/pool.py` already has a `connection()` context manager that checks out, yields, and releases with rollback.

### Steps

1. Define a `ConnectionProvider` protocol in `db/pool.py` with a single `connection()` context manager method that yields a `sqlite3.Connection`.
2. Implement `SingleConnectionProvider` — wraps a raw `sqlite3.Connection` and yields it unchanged (no rollback on exit). This preserves the current write-path semantics where the caller manages commits.
3. Verify that `ConnectionPool` already satisfies the `ConnectionProvider` protocol (it does — its `connection()` method matches). Add a `runtime_checkable` decorator if needed.
4. Write tests for both implementations: `SingleConnectionProvider` yields the same connection repeatedly; `ConnectionPool.connection()` yields connections with `check_same_thread=False`.

### Acceptance criteria

- `ConnectionProvider` protocol exists with a `connection()` context manager.
- `SingleConnectionProvider` wraps a raw connection and yields it without rollback on exit.
- `ConnectionPool` structurally satisfies `ConnectionProvider`.
- Unit tests verify both implementations.

## Phase 2: Migrate repos

Update all 27 `Sqlite*Repo` classes to accept `ConnectionProvider` instead of `sqlite3.Connection`, and wrap each method body in `with self._provider.connection() as conn:`.

### Context

Every repo follows the same pattern: `__init__(self, conn: sqlite3.Connection)` stores `self._conn`, and methods call `self._conn.execute(...)`. The migration is mechanical — replace `self._conn` with `self._provider`, and wrap method bodies in the context manager. The `Loader` class and `SqliteDatasetAssembler` also hold raw connections and need the same treatment.

### Steps

1. Update all repo `__init__` signatures from `conn: sqlite3.Connection` to `provider: ConnectionProvider`.
2. In each repo method, replace bare `self._conn.execute(...)` with `with self._provider.connection() as conn:` blocks. For simple single-statement methods, this is a one-line wrapper. For multi-statement methods (e.g., `SqlitePlayerRepo.upsert` with its INSERT/UPDATE logic), the entire method body goes inside one `with` block.
3. Update repos that call `self._conn.commit()` internally (e.g., `SqliteFeatureCandidateRepo`) to use the checked-out connection.
4. Update `Loader.__init__` to accept `ConnectionProvider` instead of `conn: sqlite3.Connection`. The `Loader.load()` method's commit/rollback calls use the checked-out connection within a `with` block that spans the entire load operation.
5. Update `SqliteDatasetAssembler` and `DatasetCatalogService` to accept `ConnectionProvider`.
6. Update `DataProfiler` and `FeatureFactory` if they hold raw connections.
7. Run the full test suite to catch any breakage (tests will fail at construction sites — that's expected and fixed in phase 3).

### Acceptance criteria

- All `Sqlite*Repo` classes accept `ConnectionProvider` in their constructor.
- Every repo method obtains its connection via `with self._provider.connection() as conn:`.
- `Loader`, `SqliteDatasetAssembler`, `DatasetCatalogService`, and any other classes that hold raw connections are migrated.
- No repo directly stores a `sqlite3.Connection` (except as yielded from the provider within a method scope).

## Phase 3: Migrate composition roots and tests

Update all call sites that construct repos — factory functions, `AnalysisContainer`, test fixtures, and test helpers — to pass a `ConnectionProvider` instead of a raw connection.

### Context

Repos are constructed in three places: `cli/factory.py` (30+ factory functions), `analysis_container.py`, and test files (723 occurrences across 83 files). Factory functions and tests currently pass a raw `sqlite3.Connection`. After phase 2, they need to pass a `ConnectionProvider`. For the write paths (factory functions), wrap the connection in `SingleConnectionProvider`. For tests, update the `conn` fixtures to yield a `SingleConnectionProvider` or add a `provider` fixture.

### Steps

1. Update all `conn` fixtures in test conftest files to also provide a `provider` fixture (a `SingleConnectionProvider` wrapping the connection). Keep the `conn` fixture for tests that need raw connection access (seeding data, direct SQL assertions).
2. Update test repo construction from `SqliteXxxRepo(conn)` to `SqliteXxxRepo(provider)` across all test files.
3. Update `tests/helpers.py` `seed_player` to accept a provider or connection as appropriate.
4. Update `AnalysisContainer.__init__` to accept `ConnectionProvider` instead of `sqlite3.Connection`.
5. Update all factory functions in `cli/factory.py` to wrap their connection in `SingleConnectionProvider` before passing to repos. Callers that need the raw connection for `commit()` keep a reference to it.
6. Update CLI commands that pass `conn` to `Loader` to pass a provider instead.
7. Run the full test suite — all tests should pass.

### Acceptance criteria

- All test files construct repos with a `ConnectionProvider`.
- `AnalysisContainer` accepts `ConnectionProvider`.
- All factory functions in `cli/factory.py` pass `SingleConnectionProvider` to repos.
- CLI commands that call `conn.commit()` still work (they hold the raw connection separately).
- Full test suite passes with no regressions.

## Phase 4: Wire pool into agent path

Replace the single-connection pattern in the agent/Discord composition roots with a `ConnectionPool`, giving each thread its own connection.

### Context

With repos now accepting `ConnectionProvider`, the agent path can receive a `ConnectionPool` instead of a `SingleConnectionProvider`. The `build_chat_context` factory currently creates one connection; it should create a pool instead. This eliminates the `check_same_thread` workaround and provides true thread-safe database access.

### Steps

1. Update `build_chat_context` in `cli/factory.py` to create a `ConnectionPool` instead of a single connection. Remove the `check_same_thread` parameter.
2. Pass the pool directly to `AnalysisContainer` (it accepts `ConnectionProvider`; `ConnectionPool` satisfies this).
3. Update `discord_cmd` and `chat_cmd` in `standalone.py` to use the new signature (no more `check_same_thread=False`).
4. Ensure `ConnectionPool.close_all()` is called on exit (the context manager should handle this).
5. Add an integration test that verifies agent tools can run from multiple threads without `ProgrammingError`.
6. Remove the now-unused `check_same_thread` parameter from `build_chat_context`.

### Acceptance criteria

- `build_chat_context` creates a `ConnectionPool`, not a single connection.
- `chat_cmd` and `discord_cmd` no longer pass `check_same_thread`.
- An integration test confirms multi-threaded tool execution works without SQLite threading errors.
- The `ConnectionPool` is properly closed when the context manager exits.

## Ordering

Phases are strictly sequential — each builds on the prior:

1. **Phase 1** (protocol) has no dependencies and can land immediately.
2. **Phase 2** (repos) depends on phase 1 for the `ConnectionProvider` type.
3. **Phase 3** (composition roots + tests) depends on phase 2 — repos won't compile with old call sites.
4. **Phase 4** (pool wiring) depends on phase 3 — `AnalysisContainer` must accept `ConnectionProvider` before it can receive a pool.

Phases 2 and 3 are the largest by line count (mechanical but wide-reaching). Phase 2 changes the repo layer; phase 3 updates all consumers. They could theoretically be combined into one phase, but splitting them keeps each diff focused: phase 2 is "repos accept providers" and phase 3 is "callers supply providers."
