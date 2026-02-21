# Resilience & Error Handling Roadmap

**Created:** 2026-02-17
**Updated:** 2026-02-17
**Status:** Proposed
**Goal:** Introduce a `Result[T, E]` type for explicit error handling at service
boundaries, add retry logic for external API calls, and improve error
diagnostics across the system.

## Motivation

The codebase uses exceptions for all error paths, but several patterns have
emerged that make failures hard to reason about:

1. **Silent failure modes.** `create_model()` silently drops unmatched kwargs,
   hiding configuration typos. Model operations raise untyped exceptions that
   callers can't distinguish without catching broad `Exception`.
2. **No retry logic.** External HTTP calls (MLB API, pybaseball) have no retry
   — a transient network failure means a failed ingest with no recovery.
3. **Ad-hoc error context.** The ingest layer manually builds `LoadLog` with
   `status="error"` in duplicated try/except blocks. This is a hand-rolled
   Result pattern without the type safety.
4. **Untyped error boundaries.** The CLI catches `UnsupportedOperation` but
   has no way to handle model failures, ingest failures, or config errors
   differently — they all bubble up as raw exceptions.

## Design: Hybrid Result + Exceptions

### Where to use `Result[T, E]`

Use `Result` at **service and orchestration boundaries** — the places where
callers need to inspect and react to errors, not just crash:

| Boundary | Current | After |
|----------|---------|-------|
| `dispatch()` | Returns `_AnyResult`, raises `UnsupportedOperation` | `Result[_AnyResult, DispatchError]` |
| `StatsLoader.load()` | Returns `LoadLog`, catches + re-raises `Exception` | `Result[LoadLog, IngestError]` — no re-raise |
| `PlayerLoader.load()` | Same pattern | `Result[LoadLog, IngestError]` |
| `ProjectionLoader.load()` | Same pattern | `Result[LoadLog, IngestError]` |
| `create_model()` | Returns `Model`, raises `KeyError` | `Result[Model, ConfigError]` |

### Where to keep exceptions

- **Programming errors**: `TypeError`, `ValueError` for invalid arguments,
  wrong types, violated preconditions. These indicate bugs, not expected
  failure modes.
- **Ecosystem boundary**: Let `sqlite3`, `httpx`, `pybaseball` throw natively.
  Convert to `Result` at the boundary wrapper, not deep inside library code.
- **CLI layer**: `typer.Exit` remains exception-based (framework requirement).

### Result type design

```python
# domain/result.py
@dataclass(frozen=True)
class Ok[T]:
    value: T

@dataclass(frozen=True)
class Err[E]:
    error: E

type Result[T, E] = Ok[T] | Err[E]
```

Minimal, no library dependency. Callers use structural pattern matching:

```python
match dispatch(operation, model, config):
    case Ok(result):
        print_result(result)
    case Err(DispatchError() as e):
        print_error(str(e))
        raise typer.Exit(code=1)
```

### Error type hierarchy

Structured error types replace the flat exception set. These are frozen
dataclasses (not exceptions) — they're values, not control flow:

```python
# domain/errors.py
@dataclass(frozen=True)
class FbmError:
    message: str

@dataclass(frozen=True)
class IngestError(FbmError):
    source_type: str
    source_detail: str
    target_table: str

@dataclass(frozen=True)
class ConfigError(FbmError):
    unrecognized_keys: tuple[str, ...] = ()

@dataclass(frozen=True)
class DispatchError(FbmError):
    model_name: str
    operation: str
```

Existing exceptions (`LeagueConfigError`, `PlayerConflictError`,
`UnsupportedOperation`) remain exceptions for now — they're used in places
where exception semantics are appropriate (validation, constraint violations).
They gain a common `FbmException` base for `except FbmException` catchability.

## Constraints

- Prefer stdlib over new libraries. The `Result` type is ~15 lines of code.
  `tenacity` is acceptable for retry logic.
- Retry behavior must be configurable (max attempts, backoff factor) but
  sensible defaults should work out of the box.
- All changes follow TDD: failing test first, then implementation.
- Migration is incremental — functions adopt `Result` one at a time, callers
  update via `match`. No big-bang rewrite.

---

## Phase 1 — Result Type & Error Types

Introduce the `Result` type and structured error hierarchy.

- Create `domain/result.py` with `Ok[T]`, `Err[E]`, and
  `type Result[T, E] = Ok[T] | Err[E]`.
- Create `domain/errors.py` with `FbmError`, `IngestError`, `ConfigError`,
  `DispatchError` as frozen dataclasses.
- Create `exceptions.py` with `FbmException(Exception)` as a base for existing
  exceptions that remain exception-based.
- Reparent existing exceptions:
  - `LeagueConfigError(FbmException)` (was `Exception`)
  - `PlayerConflictError(FbmException)` (was `Exception`)
  - `UnsupportedOperation(FbmException)` (was `Exception`)
- Add tests: `Ok`/`Err` construction, pattern matching, error type fields,
  exception inheritance chain.

## Phase 2 — Dispatcher Returns Result

Convert `dispatch()` to return `Result[_AnyResult, DispatchError]`.

- `dispatch()` returns `Err(DispatchError(...))` instead of raising
  `UnsupportedOperation` for unsupported operations.
- Model execution failures (unexpected exceptions from `.prepare()`,
  `.predict()`, etc.) are caught at the boundary and wrapped in
  `Err(DispatchError(...))`.
- Update `cli/app.py` callers to `match` on `Ok`/`Err` instead of
  `try`/`except UnsupportedOperation`. The existing `match result:` block
  in `_run_action` expands naturally:
  ```python
  match dispatch(operation, ctx.model, config):
      case Ok(PrepareResult() as r):
          print_prepare_result(r)
      case Ok(TrainResult() as r):
          print_train_result(r)
      ...
      case Err(e):
          print_error(e.message)
          raise typer.Exit(code=1)
  ```
- Update all CLI command functions that call `dispatch()`.
- Remove `UnsupportedOperation` exception class (replaced by `DispatchError`
  value).
- Update tests to assert on `Ok`/`Err` returns instead of `pytest.raises`.

## Phase 3 — Loader Returns Result

Convert the three loader classes to return `Result[LoadLog, IngestError]`.

- `StatsLoader.load()`, `PlayerLoader.load()`, `ProjectionLoader.load()`
  return `Err(IngestError(...))` instead of catching + logging + re-raising.
- The `LoadLog` with `status="error"` is still recorded to the database inside
  the `Err` path, but the error is returned, not raised.
- Callers in `cli/app.py` (ingest commands) match on the result:
  ```python
  match loader.load(**params):
      case Ok(log):
          print_load_result(log)
      case Err(e):
          print_error(e.message)
  ```
- This eliminates the duplicated try/except/build-LoadLog/re-raise pattern
  (currently repeated 6 times across the three loader classes).
- Add tests with fake sources that fail, asserting `Err(IngestError(...))`.

## Phase 4 — HTTP Retry Logic

Add retry with exponential backoff to external HTTP calls.

- Add `tenacity` to project dependencies.
- `ingest/mlb_milb_stats_source.py`: Wrap `httpx.Client.get()` calls with
  retry on `httpx.TransportError` and 5xx status codes. Default: 3 attempts,
  1s base backoff with jitter.
- `ingest/mlb_transactions_source.py`: Same retry treatment.
- Configure `httpx.Client` with explicit timeouts:
  `timeout=httpx.Timeout(10.0, connect=5.0)`.
- After retries are exhausted, the source raises — the loader (Phase 3)
  catches this and returns `Err(IngestError(...))`.
- Add unit tests using a fake HTTP transport that fails N times then succeeds.
- Log retry attempts at WARNING level.

## Phase 5 — PyBaseball Resilience

Add error handling around pybaseball calls.

- `ingest/pybaseball_source.py`: Wrap `pybaseball` calls, catching
  `requests.RequestException` and common pybaseball errors.
- Convert raw errors into exceptions with context (source type, parameters).
  The loader boundary (Phase 3) converts these into `Err(IngestError(...))`.
- Add a simple retry (2 attempts with 2s delay) for transient network errors.

## Phase 6 — Model Factory Diagnostics

Improve `create_model()` to return `Result[Model, ConfigError]`.

- When `create_model()` filters kwargs by signature inspection, collect any
  unmatched keys.
- If unmatched keys exist, return `Err(ConfigError(message=...,
  unrecognized_keys=...))`.
- Callers in `cli/app.py` match on the result instead of catching `KeyError`.
- This catches typos like `lerning_rate` that currently silently fall through
  to the model's default value.
- Add tests: unknown kwarg → `Err`; valid kwargs → `Ok`.

---

## Success Criteria

- `Result[T, E]` type exists and is used at all service boundaries listed
  above.
- Callers use `match`/`case` for error handling — no `except` for expected
  failure modes at these boundaries.
- `except FbmException` catches all remaining exception-based domain errors.
- Transient HTTP failures (network blips, 503s) are retried automatically.
- `LoadLog` entries include meaningful error context for ingestion failures.
- Typos in model hyperparameter names are caught at model creation time.
- No behavior changes for the happy path.
- Exceptions are reserved for programming errors and ecosystem boundaries.
