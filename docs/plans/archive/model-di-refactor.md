# Model Dependency Injection Refactor

## Goal

Adopt constructor injection for projection models so that all dependencies
(assembler, repos, etc.) are assembled outside the model and passed in at
construction time. This eliminates the zero-arg constructor constraint in the
registry, removes the assembler-creation fallback inside `MarcelModel.predict`,
and makes all operation protocol signatures uniform (`method(config) -> Result`).

No external DI library is introduced — the project continues with hand-rolled
constructor injection and a centralized composition root, consistent with
existing patterns in `RunManager`, `ProjectionEvaluator`, and `StatsLoader`.

## Phases

### Phase 1 — Uniform protocol signatures

Remove `assembler` from operation method signatures so every protocol method
takes only `(self, config: ModelConfig) -> SomeResult`.

- Change `Preparable.prepare` from `(config, assembler)` to `(config)`.
- Update `_dispatcher.py`: remove `_ASSEMBLER_OPERATIONS` and the branching
  that passes `assembler` to specific operations. All operations call
  `method(config)` uniformly.
- Update `MarcelModel`: accept `DatasetAssembler` as a constructor parameter,
  store it as `self._assembler`, and use it in `prepare` and `predict`.
- Remove the `create_connection` / `SqliteDatasetAssembler` fallback inside
  `MarcelModel.predict`.
- Update all existing tests that pass `assembler` as a method argument to
  instead inject it via the constructor.

### Phase 2 — Registry returns classes, not instances

Change the registry so it stores and returns model *classes* rather than
instantiating them with a zero-arg call.

- `registry.get(name)` returns the class (or a factory callable), not an
  instance.
- Introduce a composition-root helper (e.g. `create_model(name, ...)` in
  `cli/factory.py` or inline in `app.py`) that looks up the class from the
  registry and instantiates it with the appropriate dependencies.
- Update `_dispatcher.dispatch` to accept a `ProjectionModel` instance instead
  of a `model_name` string — the caller (CLI) is responsible for construction.
- Update CLI commands (`_run_action`, `train`, etc.) to use the composition
  root for model construction.
- Update tests that call `registry.get()` expecting an instance.

### Phase 3 — Composition root and dispatcher cleanup

Consolidate all dependency wiring into a single composition root so the full
object graph is visible in one place.

- Create a small factory or context object that owns the database connection
  and builds all collaborators (assembler, repos, run manager) needed by a
  given CLI invocation.
- Simplify `_run_action` and the `train` command to delegate to this factory
  instead of manually constructing deps inline.
- Remove any remaining `create_connection` calls from model code — models
  should never touch infrastructure directly.
- Verify the dispatcher is now a thin protocol-check + method-call with no
  dependency-routing logic.

### Phase 4 — Validate and clean up

- Run full quality gate: `pytest`, `ruff check`, `ty check`.
- Confirm all tests pass with constructor-injected fakes (no monkeypatch for
  collaborators).
- Verify that adding a hypothetical new model with new dependencies requires
  only: (1) defining the class with constructor params, (2) registering it,
  (3) adding a branch in the composition root. No dispatcher or protocol
  changes needed.

## Out of scope

- Introducing a third-party DI container library.
- Async or scope-based lifecycle management.
- Refactoring non-model services (they already use constructor injection).
