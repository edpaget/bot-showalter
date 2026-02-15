# Model Registry

The model registry manages projection models, run tracking, evaluation, and comparison. It unifies first-party models (trained in-house) and third-party projections (imported from external systems) under a single evaluation framework built on top of the [data layer](data-layer.md).

## Core concept: projections are the common currency

Every projection system — Marcel, XGBoost, Steamer, ZiPS — writes rows to the `projection` table. Evaluation and comparison operate at this level and don't care about provenance. The difference between first-party and third-party is how the projections got there:

| | First-party (Marcel, XGBoost, ...) | Third-party (Steamer, ZiPS, ...) |
|---|---|---|
| Produces projection rows | Yes | Yes |
| Evaluable against actuals | Yes | Yes |
| Comparable to other systems | Yes | Yes |
| Has a `model_run` record | Yes | No |
| Has serialized artifacts | Maybe (ML models yes, statistical no) | No |
| In the code registry | Yes | No |

---

## Package overview

```
src/fantasy_baseball_manager/
    models/              # Registry, protocols, run management
    domain/              # ModelRunRecord, ArtifactType, evaluation types
    services/            # ProjectionEvaluator
    cli/                 # CLI commands (train, import, eval, compare, runs)
    config.py            # TOML config with version/tags support
```

---

## `models/` — Registry and run management

### Model registry

The registry is a name-to-factory lookup for first-party models only. It does not know about third-party systems or versions.

```python
from fantasy_baseball_manager.models.registry import get, list_models, register

# List registered models
list_models()  # ["marcel"]

# Get a model class (not an instance — see docs/model-di.md)
cls = get("marcel")

# Instantiate via the composition root or directly with deps
model = cls(assembler=assembler)
model.name             # "marcel"
model.description      # "Marcel the Monkey Forecasting System"
model.supported_operations  # frozenset({"prepare", "train", "evaluate"})
model.artifact_type    # "none"
```

### Model protocol

Every registered model implements `Model` and one or more operation protocols:

```python
@runtime_checkable
class Model(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def supported_operations(self) -> frozenset[str]: ...
    @property
    def artifact_type(self) -> str: ...  # "none", "file", or "directory"
```

Operation protocols: `Trainable`, `Evaluable`, `Preparable`, `Predictable`, `FineTunable`, `Ablatable`. Each defines a single method (e.g. `train(config) -> TrainResult`).

### ArtifactType

Models declare what kind of artifact they produce:

| Value | Meaning | Example |
|---|---|---|
| `none` | Statistical model, reproducible from config + code | Marcel |
| `file` | Single serialized file | XGBoost (`.joblib`) |
| `directory` | Directory of files | TensorFlow SavedModel |

### RunManager

Orchestrates the lifecycle of a first-party model run. Created by the CLI when `--version` is provided on `fbm train`.

```python
from fantasy_baseball_manager.models.run_manager import RunManager, RunContext

mgr = RunManager(model_run_repo=repo, artifacts_root=Path("./artifacts"))

# Begin a run — creates artifact directory if needed
ctx = mgr.begin_run(model, config)
ctx.run_dir    # Path to write artifacts
ctx.system     # Model name
ctx.version    # Config version

# Log metrics during training
ctx.log_metric("rmse", 0.42)

# Finalize — persists the ModelRunRecord to the database
mgr.finalize_run(ctx, config)

# Delete — removes DB record and artifact directory
mgr.delete_run("marcel", "v1")
```

The dispatcher calls `begin_run` before delegating to the model's `train()` method and `finalize_run` after it returns. This keeps run-tracking logic out of individual models.

### RunContext

Provides the model with a managed directory and metric logging:

| Property/Method | Description |
|---|---|
| `run_dir` | `Path` where artifacts should be written |
| `system` | Model name |
| `version` | Config version string |
| `metrics` | Copy of logged metrics dict |
| `log_metric(key, value)` | Record a metric (overwrites on duplicate key) |

---

## `domain/` — Types

### ModelRunRecord

```python
@dataclass(frozen=True)
class ModelRunRecord:
    system: str                              # Matches projection.system
    version: str                             # Matches projection.version
    config_json: dict[str, Any]              # Full ModelConfig snapshot
    artifact_type: str                       # "none", "file", "directory"
    created_at: str                          # ISO 8601
    train_dataset_id: int | None = None
    validation_dataset_id: int | None = None
    holdout_dataset_id: int | None = None
    metrics_json: dict[str, Any] | None = None
    artifact_path: str | None = None
    git_commit: str | None = None            # HEAD at time of run
    tags_json: dict[str, str] | None = None  # Arbitrary metadata
    id: int | None = None
```

### Evaluation types

```python
@dataclass(frozen=True)
class StatMetrics:
    rmse: float
    mae: float
    correlation: float
    n: int                  # Number of players evaluated

@dataclass(frozen=True)
class SystemMetrics:
    system: str
    version: str
    source_type: str        # "first_party" or "third_party"
    metrics: dict[str, StatMetrics]

@dataclass(frozen=True)
class ComparisonResult:
    season: int
    stats: list[str]
    systems: list[SystemMetrics]
```

---

## `repos/` — ModelRunRepo

Protocol and SQLite implementation following the same patterns as other repos in the data layer.

```python
@runtime_checkable
class ModelRunRepo(Protocol):
    def upsert(self, record: ModelRunRecord) -> int: ...
    def get(self, system: str, version: str) -> ModelRunRecord | None: ...
    def list(self, system: str | None = None) -> list[ModelRunRecord]: ...
    def delete(self, system: str, version: str) -> None: ...
```

```python
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo

repo = SqliteModelRunRepo(conn)
runs = repo.list(system="marcel")   # All marcel runs, newest first
run = repo.get("marcel", "v1")      # Single run by system+version
repo.delete("marcel", "v1")         # Remove a run record
```

The `model_run` table uses a `UNIQUE(system, version)` constraint. `upsert` replaces existing records on conflict.

---

## `services/` — ProjectionEvaluator

Evaluates any projection system against actual stats. Works identically for first-party and third-party projections.

```python
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator

evaluator = ProjectionEvaluator(projection_repo, batting_repo, pitching_repo)

# Evaluate a single system
metrics = evaluator.evaluate("marcel", "v1", season=2025, stats=["hr", "avg"])
# metrics.metrics["hr"].rmse, metrics.metrics["hr"].mae, etc.

# Compare multiple systems
result = evaluator.compare(
    [("marcel", "v1"), ("steamer", "2025.1"), ("zips", "2025")],
    season=2025,
)
# result.systems[0].metrics["hr"].rmse vs result.systems[1].metrics["hr"].rmse
```

The evaluator pulls projected stats from the `projection` table and actuals from `batting_stats`/`pitching_stats`, then computes RMSE, MAE, and Pearson correlation per stat.

---

## Configuration

### ModelConfig

```python
@dataclass(frozen=True)
class ModelConfig:
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    seasons: list[int] = field(default_factory=list)
    model_params: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    version: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
```

### TOML config

```toml
[common]
data_dir = "./data"
artifacts_dir = "./artifacts"
seasons = [2021, 2022, 2023, 2024, 2025]

[models.marcel]
version = "2026.1"

[models.marcel.params]
weights = [5, 4, 3]
regression_pct = 0.4

[models.marcel.tags]
experiment = "baseline"
```

### CLI overrides

CLI options override TOML values. `--version` replaces the TOML version. `--tag key=value` merges on top of TOML tags (CLI wins on key conflict).

```python
from fantasy_baseball_manager.config import load_config

config = load_config(
    model_name="marcel",
    version="v2",                    # Overrides TOML version
    tags={"env": "prod"},            # Merges with TOML tags
    seasons=[2023, 2024],            # Overrides TOML seasons
)
```

---

## CLI commands

### Training with run tracking

```bash
# Train without tracking (no run record created)
fbm train marcel

# Train with version — automatically records a model run
fbm train marcel --version v1

# Train with version and tags
fbm train marcel --version v1 --tag env=test --tag experiment=baseline

# Override seasons from CLI
fbm train marcel --version v1 --season 2023 --season 2024
```

When `--version` is provided, the CLI creates a `RunManager` that records the run in the `model_run` table, capturing the config snapshot, git commit hash, and any metrics logged during training.

### Third-party projection import

```bash
fbm import steamer projections.csv --version 2026 --player-type batter --season 2025
fbm import zips pitching.csv --version 2026 --player-type pitcher --season 2025
```

Imports write to the `projection` table with `source_type = "third_party"` and create a `load_log` entry.

### Evaluation and comparison

```bash
# Evaluate a single system (works for first-party and third-party)
fbm eval marcel --version v1 --season 2025
fbm eval steamer --version 2026 --season 2025

# Filter to specific stats
fbm eval marcel --version v1 --season 2025 --stat hr --stat avg

# Compare multiple systems side by side
fbm compare marcel/v1 steamer/2026 zips/2026 --season 2025
```

### Run management

```bash
# List all recorded runs
fbm runs list

# Filter by model
fbm runs list --model marcel

# Show full details (config, metrics, git commit, tags)
fbm runs show marcel/v1

# Delete a run and its artifacts
fbm runs delete marcel/v1 --yes
```

### Other model commands

```bash
fbm list                  # List registered models
fbm info marcel           # Show model metadata and supported operations
fbm features marcel       # List declared features
fbm prepare marcel        # Prepare training data
fbm evaluate marcel       # Evaluate using model's built-in evaluation
fbm predict marcel        # Generate predictions
fbm ablate marcel         # Run ablation study
```

---

## Traceability chains

### First-party models

```
projection row
  -> system + version
    -> model_run
      -> config_json (exact parameters used)
      -> train_dataset_id -> dataset -> ds_{id} table (exact training rows)
      -> artifact_path (serialized model, if any)
      -> git_commit (code version)
```

### Third-party projections

```
projection row
  -> loaded_at timestamp
    -> load_log (source file, import time, row count)
```

---

## Schema

The `model_run` table lives in `stats.db` alongside other data-layer tables.

```sql
CREATE TABLE model_run (
    id                    INTEGER PRIMARY KEY,
    system                TEXT NOT NULL,
    version               TEXT NOT NULL,
    train_dataset_id      INTEGER REFERENCES dataset(id),
    validation_dataset_id INTEGER REFERENCES dataset(id),
    holdout_dataset_id    INTEGER REFERENCES dataset(id),
    config_json           TEXT NOT NULL,
    metrics_json          TEXT,
    artifact_type         TEXT NOT NULL,
    artifact_path         TEXT,
    git_commit            TEXT,
    tags_json             TEXT,
    created_at            TEXT NOT NULL,
    UNIQUE(system, version)
);
```

The `projection` table includes a `source_type` column (`"first_party"` or `"third_party"`) to distinguish provenance.

---

## Testing

Run tracking and evaluation use the same protocol-based testing pattern as the data layer. Inject `FakeModelRunRepo` via constructor rather than patching:

```python
class FakeModelRunRepo:
    def __init__(self) -> None:
        self._records: list[ModelRunRecord] = []

    def upsert(self, record: ModelRunRecord) -> int: ...
    def get(self, system: str, version: str) -> ModelRunRecord | None: ...
    def list(self, system: str | None = None) -> list[ModelRunRecord]: ...
    def delete(self, system: str, version: str) -> None: ...
```

CLI tests use `typer.testing.CliRunner` with monkeypatched `create_connection` to inject in-memory databases.
