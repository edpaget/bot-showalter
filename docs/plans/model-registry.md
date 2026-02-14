# Model Registry Roadmap

## Goal

Evolve the current name→factory model registry so that every projection — whether produced by a first-party model run or imported from a third-party system — is **versionable**, **evaluable**, and **comparable**. First-party runs are additionally **traceable** to the config and datasets that produced them.

This work builds on the data layer (see `data-layer.md`). It should be implemented **after** the data-layer roadmap is complete, so we build directly against SQLite rather than maintaining a throwaway interim store.

---

## Problem Statement

The current registry (`models/registry.py`) maps a model name to a factory callable. It has no concept of:

- **Versions** — there's no way to distinguish two runs of the same model with different configurations.
- **Artifact management** — `TrainResult` carries an `artifacts_path` string but nothing manages what goes there, how it's organized, or how to retrieve it later.
- **Provenance** — no link from a model run back to the config that produced it, the datasets it consumed, or the code version that ran.
- **Third-party projections** — Steamer, ZiPS, ATC, and other external systems produce projections we import via CSV. We want to evaluate and compare these against actuals (and against our own models) using the same tools, even though we never "run" them.
- **Statistical models** — Marcel doesn't produce a serialized artifact. Its "model" is the algorithm plus its config. This should be a first-class case, not an exception.

---

## Core Insight: Unify on Projections, Differentiate on Provenance

The data-layer plan already defines a `projection` table — one row per player per system per version. Every projection system, first-party or third-party, writes to this table. This is where evaluation and comparison happen.

The difference between first- and third-party is **provenance**, not output format:

| | First-party (Marcel, XGBoost, ...) | Third-party (Steamer, ZiPS, ...) |
|---|---|---|
| Produces projection rows | Yes | Yes |
| Evaluable against actuals | Yes | Yes |
| Comparable to other systems | Yes | Yes |
| Has a `model_run` record | Yes — links to config, datasets, metrics, artifacts | No — has a `load_log` entry tracking the import |
| Has serialized artifacts | Maybe (ML models yes, statistical no) | No |
| In the code registry | Yes | No (no code to register) |

Evaluation and comparison work at the **projection** level. Provenance is an optional deeper layer you can drill into for first-party models.

---

## Design

### Projection Sources

Every projection system is either:

- **`first_party`** — we run it. A code model in the registry produces projections. It may or may not produce serialized artifacts. It always has a `model_run` record linking to config, datasets, and metrics.
- **`third_party`** — we import it. Projections arrive via CSV (or API). There's no model run, but the `load_log` records when and how the data was imported.

The `projection` table's `system` column already captures this distinction implicitly (e.g. `"marcel"` vs `"steamer"`). We add a `source_type` column to make it explicit and queryable:

```sql
ALTER TABLE projection ADD COLUMN source_type TEXT NOT NULL DEFAULT 'first_party';
-- values: 'first_party', 'third_party'
```

### Model Run Tracking (first-party only)

The data-layer plan defines a `model_run` table. We extend it to capture full provenance:

```sql
CREATE TABLE model_run (
    id              INTEGER PRIMARY KEY,
    system          TEXT NOT NULL,       -- matches projection.system
    version         TEXT NOT NULL,       -- matches projection.version
    train_dataset_id      INTEGER REFERENCES dataset(id),
    validation_dataset_id INTEGER REFERENCES dataset(id),
    holdout_dataset_id    INTEGER REFERENCES dataset(id),
    config_json     TEXT NOT NULL,       -- full ModelConfig snapshot
    metrics_json    TEXT,                -- evaluation metrics (RMSE, MAE, etc.)
    artifact_type   TEXT NOT NULL,       -- 'none', 'file', 'directory'
    artifact_path   TEXT,                -- relative path under artifacts root
    git_commit      TEXT,                -- HEAD at time of run
    tags_json       TEXT,                -- arbitrary metadata
    created_at      TEXT NOT NULL,       -- ISO 8601
    UNIQUE(system, version)
);
```

This closes the full traceability chain for first-party models:

```
projection row
  → system + version
    → model_run
      → config_json (exact parameters used)
      → train_dataset_id → dataset → ds_{id} table (exact training rows)
      → artifact_path (serialized model, if any)
      → git_commit (code version)
```

Third-party projections skip this chain — their traceability goes through `load_log` instead:

```
projection row
  → loaded_at timestamp
    → load_log (source file, import time, row count)
```

### Artifact Storage

Artifacts live under a configurable root (default `~/.fbm/artifacts/`):

```
artifacts/
  marcel/
    2026.1/
      (empty — statistical model, no serialized artifact)
    2026.2/
      (empty)
  xgb-batter/
    v3/
      model.joblib
    v4/
      model.joblib
```

The `model_run` table is the source of truth for provenance. The filesystem is just a durable store for serialized artifacts. Models that produce no artifact (`artifact_type = 'none'`) simply have no directory.

### ArtifactType Enum

```python
class ArtifactType(Enum):
    NONE = "none"           # Statistical models — reproducible from config + code
    FILE = "file"           # Single serialized file (pickle, joblib, ONNX)
    DIRECTORY = "directory"  # Directory of files (e.g. TF SavedModel)
```

Models declare their artifact type via a property on `ProjectionModel`:

```python
@runtime_checkable
class ProjectionModel(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def supported_operations(self) -> frozenset[str]: ...
    @property
    def artifact_type(self) -> ArtifactType: ...  # NEW
```

### ModelRunRepo

A repository for the `model_run` table, following the same patterns as other repos in the data layer:

```python
@runtime_checkable
class ModelRunRepo(Protocol):
    def save(self, record: ModelRunRecord) -> int: ...
    def get(self, system: str, version: str) -> ModelRunRecord | None: ...
    def list(self, system: str | None = None) -> list[ModelRunRecord]: ...
    def delete(self, system: str, version: str) -> None: ...
```

### RunManager

Orchestrates the lifecycle of a first-party model run:

```python
class RunManager:
    def __init__(
        self,
        model_run_repo: ModelRunRepo,
        artifacts_root: Path,
    ) -> None: ...

    def begin_run(self, model: ProjectionModel, config: ModelConfig) -> RunContext: ...
    def finalize_run(self, context: RunContext, metrics: dict[str, float]) -> ModelRunRecord: ...
```

`RunContext` gives the model a managed directory to write artifacts and a way to log metrics:

```python
class RunContext:
    @property
    def run_dir(self) -> Path: ...
    def log_metric(self, key: str, value: float) -> None: ...
```

The `begin_run` / `finalize_run` pattern keeps run-tracking logic out of individual models. The dispatcher calls `begin_run` before delegating to the model and `finalize_run` after it returns.

### Projection Evaluator

A service that evaluates **any** projection system against actuals — first-party or third-party:

```python
class ProjectionEvaluator:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
    ) -> None: ...

    def evaluate(
        self,
        system: str,
        version: str,
        season: int,
        stats: list[str] | None = None,
    ) -> EvalResult: ...

    def compare(
        self,
        systems: list[tuple[str, str]],  # (system, version) pairs
        season: int,
        stats: list[str] | None = None,
    ) -> ComparisonResult: ...
```

This is the key unifier. It doesn't care whether the projection came from Marcel, XGBoost, or Steamer. It pulls projected stats from the `projection` table, pulls actuals from `batting_stats`/`pitching_stats`, and computes error metrics (RMSE, MAE, correlation, etc.) per stat.

```python
@dataclass(frozen=True)
class ComparisonResult:
    season: int
    stats: list[str]
    systems: list[SystemMetrics]

@dataclass(frozen=True)
class SystemMetrics:
    system: str
    version: str
    source_type: str              # 'first_party' or 'third_party'
    metrics: dict[str, StatMetrics]  # stat name → error metrics

@dataclass(frozen=True)
class StatMetrics:
    rmse: float
    mae: float
    correlation: float
    n: int                        # number of players evaluated
```

### Integration with the Registry

The registry remains a name→factory lookup for **first-party models only**. It doesn't know about third-party systems. It doesn't become version-aware — versions are a property of runs and projections, not of code.

```
registry.get("marcel")                    → MarcelModel instance (code)
model_run_repo.list(system="marcel")      → [RunRecord(v=2026.1), ...]
projection_repo.list(system="steamer")    → [projections...] (no model_run)
evaluator.compare([("marcel","2026.1"), ("steamer","2026")], season=2025)  → ComparisonResult
```

### Config Changes

`ModelConfig` gains `version` and `tags`:

```python
@dataclass(frozen=True)
class ModelConfig:
    data_dir: str = "./data"
    artifacts_dir: str = "./artifacts"
    seasons: list[int] = field(default_factory=list)
    model_params: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
    version: str | None = None                          # NEW
    tags: dict[str, str] = field(default_factory=dict)  # NEW
```

TOML:

```toml
[models.marcel]
version = "2026.1"

[models.marcel.params]
weights = [5, 4, 3]
regression_pct = 0.4

[models.marcel.tags]
experiment = "baseline"
```

### CLI Additions

```
# First-party model runs (existing commands, new --version flag)
fbm train marcel --version 2026.1 --season 2023 --season 2024 --season 2025

# Third-party import (new command)
fbm import steamer projections.csv --version 2026 --player-type batter

# Run management (first-party only)
fbm runs list                            # All first-party runs
fbm runs list --model marcel             # Marcel runs only
fbm runs show marcel/2026.1              # Full provenance

# Evaluation and comparison (works for any projection source)
fbm eval marcel --version 2026.1 --season 2025
fbm eval steamer --version 2026 --season 2025
fbm compare marcel/2026.1 steamer/2026 zips/2026 --season 2025
```

---

## Dependency on Data Layer

This roadmap depends on the data-layer roadmap being complete (or at least through Phase 6). Specifically it requires:

- `projection` table and `ProjectionRepo` (data-layer Phase 5)
- `feature_set`, `dataset` tables (data-layer Phase 6)
- `batting_stats`, `pitching_stats` tables and repos (data-layer Phase 3)
- `load_log` table and repo (data-layer Phase 2)
- Connection infrastructure, migration runner (data-layer Phase 1)

By building on the data layer rather than alongside it, we avoid maintaining a throwaway `FileRunStore` and get queryability (sort runs by RMSE, filter by date, etc.) from day one.

---

## Phases

All phases below assume the data-layer roadmap is complete.

### Phase 1 — Core types and model_run table

- `ArtifactType` enum
- `ModelRunRecord` frozen dataclass
- `model_run` table schema (migration extending data-layer's `model_run` with `config_json`, `artifact_type`, `artifact_path`, `git_commit`, `tags_json`)
- Add `source_type` column to `projection` table (migration)
- `ModelRunRepo` protocol and SQLite implementation
- Tests with in-memory SQLite

### Phase 2 — RunManager and RunContext

- `RunContext` class (managed run directory, metric logging)
- `RunManager` class (begin/finalize lifecycle, git commit capture, delegates to `ModelRunRepo`)
- Update `ModelConfig` with `version` and `tags`
- Add `artifact_type` to `ProjectionModel` protocol
- Update Marcel: `artifact_type = ArtifactType.NONE`
- Wire `RunManager` into the dispatcher so `train` automatically records a model run
- Update existing tests

### Phase 3 — Third-party projection import

- `fbm import` CLI command: reads a CSV, maps columns, writes to `projection` table with `source_type = 'third_party'`
- Column mapping config for common systems (Steamer, ZiPS, ATC) — reuses `ingest/column_maps.py` from the data layer
- Import recorded in `load_log`
- Tests with sample CSV fixtures

### Phase 4 — Projection evaluation and comparison

- `ProjectionEvaluator` service (compares projections to actuals, computes RMSE/MAE/correlation)
- `ComparisonResult` and `StatMetrics` dataclasses
- `fbm eval` and `fbm compare` CLI commands
- Output formatting (table of metrics across systems)
- Works identically for first-party and third-party projections
- Tests with known projection/actual pairs

### Phase 5 — Run management CLI

- `fbm runs list`, `fbm runs show`, `fbm runs delete` commands
- `--version` and `--tag` options on `fbm train`
- TOML config support for `version` and `tags`
- Artifact directory cleanup on `runs delete`
- Integration tests
