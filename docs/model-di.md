# Model Dependency Injection

Projection models use constructor injection for all dependencies. There is no DI framework — dependencies are assembled by hand in a centralized composition root, consistent with the rest of the codebase (repos, evaluators, loaders).

## Key invariants

- **Models never touch infrastructure directly.** No `create_connection`, no file I/O, no config loading. Everything arrives via the constructor.
- **Operation signatures are uniform.** Every operation protocol method takes `(self, config: ModelConfig) -> SomeResult`. No operation-specific parameters.
- **The registry stores classes, not instances.** `registry.get(name)` returns a class (or callable). The caller is responsible for construction.
- **The dispatcher is a thin protocol check + method call.** It has no dependency-routing logic.

## How the pieces fit together

```
CLI command
  -> build_model_context(model_name, config)    # composition root
       -> create_connection(...)                 # open DB
       -> SqliteDatasetAssembler(conn)           # build collaborators
       -> create_model(name, assembler=...)      # registry lookup + construct
       -> RunManager(repo, artifacts_root)       # if --version provided
       -> yield ModelContext(conn, model, run_manager)
  -> dispatch(operation, model, config)          # protocol check + call
```

### Registry

```python
from fantasy_baseball_manager.models.registry import register, get, list_models

@register("marcel")
class MarcelModel:
    def __init__(self, assembler: DatasetAssembler | None = None) -> None: ...

# Returns the class, not an instance
cls = get("marcel")
```

### Composition root

`cli/factory.py` contains `build_model_context`, a context manager that owns the DB connection and wires all collaborators:

```python
@contextmanager
def build_model_context(model_name: str, config: ModelConfig) -> Iterator[ModelContext]:
    conn = create_connection(Path(config.data_dir) / "fbm.db")
    try:
        assembler = SqliteDatasetAssembler(conn)
        model = create_model(model_name, assembler=assembler)

        run_manager = None
        if config.version is not None:
            repo = SqliteModelRunRepo(conn)
            run_manager = RunManager(model_run_repo=repo, artifacts_root=...)

        yield ModelContext(conn=conn, model=model, run_manager=run_manager)
    finally:
        conn.close()
```

`create_model` uses `inspect.signature` to forward only kwargs matching the constructor, so models that don't need a given collaborator simply don't declare it.

### Dispatcher

`cli/_dispatcher.py` maps operation names to `(protocol, method_name)` pairs. It checks `isinstance(model, protocol)`, then calls `getattr(model, method_name)(config)`. No dependency routing.

## Adding a new model

Three things are needed — no changes to protocols, dispatcher, or existing models:

1. **Define the class** with constructor parameters for its dependencies:

   ```python
   @register("xgboost")
   class XGBoostModel:
       def __init__(self, assembler: DatasetAssembler | None = None,
                    scorer: XGBScorer | None = None) -> None:
           self._assembler = assembler
           self._scorer = scorer

       def train(self, config: ModelConfig) -> TrainResult: ...
   ```

2. **Register it** via the `@register` decorator (shown above).

3. **Wire dependencies in the composition root** by adding a branch in `build_model_context` or extending the kwargs passed to `create_model`.

## Testing

Models are tested by injecting fakes via constructors — no `monkeypatch` or `mock.patch` for collaborators. Since dependencies are typed with `Protocol`, the type checker verifies that test doubles match the real interface:

```python
class FakeAssembler:
    def get_or_materialize(self, feature_set): ...
    def read(self, handle): ...

model = MarcelModel(assembler=FakeAssembler())
result = model.prepare(config)
```

CLI-level tests monkeypatch `create_connection` (global state) to inject in-memory databases, then exercise the full `build_model_context` -> `dispatch` flow.
