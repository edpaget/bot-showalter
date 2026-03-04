# Experiment System: Generalization Analysis

This document maps which parts of the experiment infrastructure are model-agnostic, which are tied to specific model families, and what it would take to generalize the system for use with non-GBM models.

## Current architecture

The experiment system has four layers, each with a different level of coupling:

```
┌─────────────────────────────────────────────────┐
│  /experiment skill            statcast-gbm only  │
├─────────────────────────────────────────────────┤
│  CLI commands (quick-eval,    statcast-gbm only  │
│  marginal-value, residuals,   (isinstance gate)  │
│  compare-features, etc.)                         │
├─────────────────────────────────────────────────┤
│  Services (quick_eval(),      GBM-family only    │
│  marginal_value(),            (imports           │
│  compare_feature_sets())       gbm_training)     │
├─────────────────────────────────────────────────┤
│  Domain + Repo + Summary      fully generic      │
│  (Experiment, TargetResult,                      │
│   SqliteExperimentRepo,                          │
│   summarize_exploration)                         │
└─────────────────────────────────────────────────┘
```

## Layer-by-layer breakdown

### Layer 1: Domain, repository, and summary — fully generic

These components have no model-specific types or assumptions:

- **`domain/experiment.py`** — `Experiment`, `TargetResult`, `ExplorationSummary` are pure data containers. The `model` field is a string; `target_results` is a `dict[str, TargetResult]` with no constraints on target names.
- **`repos/experiment_repo.py`** — `SqliteExperimentRepo` stores experiments as JSON keyed by `(model, player_type)`. Supports filtering, searching by feature/target/tag, and finding best results. Works with any model name.
- **`services/experiment_summary.py`** — `summarize_exploration()` accepts a `(model, player_type)` pair and aggregates experiment history. No model-specific logic.

**Usable today with any model type.** Any model that can express its results as per-target RMSE deltas can log to and query the experiment journal.

### Layer 2: Quick-eval services — GBM-family coupled

`services/quick_eval.py` contains `quick_eval()`, `marginal_value()`, and `compare_feature_sets()`. Their function signatures are generic (they accept feature columns, targets, and row dicts), but they import `extract_features`, `extract_targets`, and `fit_models` from `models/gbm_training.py`, which:

- Trains `HistGradientBoostingRegressor` models
- Has hardcoded derived-target logic (iso, hr_per_9, babip formulas in `extract_targets`)
- Returns sklearn model objects

These services work for any model that uses the same `gbm_training` infrastructure (currently only statcast-gbm), but they cannot be used with a fundamentally different training pipeline without modification.

### Layer 3: CLI commands — statcast-gbm gatekept

`quick_eval.py`, `marginal_value.py`, `residuals.py`, `compare_features.py`, `validate.py`, and `report.py` all have:

```python
if not isinstance(model_instance, _StatcastGBMBase):
    print_error("not a StatcastGBM model")
    raise typer.Exit(code=1)
```

They also directly access `model_instance._batter_columns`, `._pitcher_columns`, `._batter_training_set_builder()`, and import `BATTER_TARGETS` / `PITCHER_TARGETS` from the statcast-gbm module.

### Layer 4: /experiment skill — statcast-gbm specific

The skill's documentation, target lists, available statcast columns, SQL expression examples, and workflow are all written for statcast-gbm.

## Model family taxonomy

Not all models benefit from the same experimentation tools. Here's what applies where:

### Tabular regressors (GBM, random forest, ridge, elastic net, etc.)

These models share the same fundamental structure: a feature matrix of numeric columns, one or more regression targets, and fast retraining.

| Component | Applicable? | Notes |
|---|---|---|
| Experiment journal | Yes | Log hypotheses, deltas, and conclusions for any model |
| Exploration summary | Yes | Aggregate experiment history per model/player-type |
| Feature marginal value | Yes | Add/remove a column, retrain, compare RMSE — the core loop works identically |
| Quick-eval | Yes | Single-target train-and-score is model-agnostic in concept |
| Feature correlation screening | Yes | Pearson/Spearman of candidate vs target is model-independent |
| Temporal stability profiling | Yes | Year-over-year correlation of a feature is model-independent |
| Residual analysis | Yes | Worst-miss and cohort-bias diagnostics work on any model's predictions |
| Permutation importance | Yes | Works on any fitted model with a `.predict()` method |
| Compare feature sets | Yes | A/B feature set comparison on identical data splits |

**The experiment loop transfers almost entirely.** The main constraint is retraining speed — GBMs train in seconds, which makes the rapid iteration cycle practical. Ridge/elastic net would be even faster. Random forests would be comparable.

### Ensemble / composite models

These combine multiple sub-models (e.g., a weighted blend of statcast-gbm + steamer + ATC).

| Component | Applicable? | Notes |
|---|---|---|
| Experiment journal | Yes | Log weight changes, blend strategy experiments |
| Quick-eval / marginal value | No | Feature engineering doesn't apply — the "features" are sub-model outputs |
| Residual analysis | Yes | Diagnose where the ensemble struggles vs individual components |
| Compare feature sets | Partially | Could compare "which sub-models to include" rather than columns |

The experiment loop concept applies but the **unit of experimentation changes** from "add/remove a feature column" to "adjust blend weights" or "include/exclude a sub-model." The journal is reusable; the fast-feedback tools are not.

### Neural networks / deep learning

If we ever added a neural model (embeddings, transformers, etc.):

| Component | Applicable? | Notes |
|---|---|---|
| Experiment journal | Yes | Log architecture changes, hyperparameter experiments |
| Quick-eval / marginal value | No | Retraining is too slow for rapid iteration; feature engineering is less relevant since the model learns representations |
| Residual analysis | Yes | Diagnose systematic prediction errors |
| Correlation screening | Partially | Can screen raw inputs, but learned features dominate |

The experiment **journal and workflow pattern** (hypothesize → test → log → iterate) transfer. The **fast-feedback tools do not** — you'd need different evaluation mechanisms (probing frozen embeddings, ablation studies, learning curve analysis).

### Lookup / projection systems (e.g., Marcel, Steamer-style)

These aren't trained in the ML sense — they use formulas, aging curves, and regression to the mean.

| Component | Applicable? | Notes |
|---|---|---|
| Experiment journal | Yes | Log formula changes, aging curve adjustments |
| Quick-eval / marginal value | No | No feature matrix to manipulate |
| Residual analysis | Yes | Where does the formula systematically miss? |

## What generalization would look like

### Phase 1: Extract a Protocol for experimentable models

The minimum viable generalization. Define what a model needs to expose to participate in the experiment loop:

```python
@runtime_checkable
class Experimentable(Protocol):
    """A model that supports feature-engineering experiments."""

    def feature_columns(self, player_type: str) -> list[str]:
        """Current feature set for the given player type."""
        ...

    def targets(self, player_type: str) -> list[str]:
        """Target stats for the given player type."""
        ...

    def training_data(
        self, player_type: str, seasons: list[int]
    ) -> dict[int, list[dict[str, Any]]]:
        """Materialized training rows grouped by season."""
        ...
```

The CLI commands would replace `isinstance(_StatcastGBMBase)` with `isinstance(model, Experimentable)`, and access feature columns, targets, and data through the protocol instead of private attributes.

**Scope:** ~6 CLI files, ~10-15 lines each. The services and domain layer require no changes.

### Phase 2: Abstract the training backend

Currently `quick_eval()` and `marginal_value()` import `fit_models` from `gbm_training`, which returns `HistGradientBoostingRegressor` objects. To support other model families:

```python
class TrainingBackend(Protocol):
    def fit(
        self,
        X: list[list[float]],
        targets: dict[str, TargetVector],
        params: dict[str, Any],
    ) -> FittedModels:
        ...

class FittedModels(Protocol):
    def predict(self, target: str, X: list[list[float]]) -> np.ndarray: ...
```

Then `quick_eval()` and `marginal_value()` would accept a `TrainingBackend` parameter instead of importing `gbm_training` directly. Each model family would provide its own backend implementation.

**Scope:** Refactor `services/quick_eval.py` to accept a backend parameter; implement `GBMTrainingBackend` wrapping the current `gbm_training` functions. New model families implement their own backend.

### Phase 3: Generalize the skill

The `/experiment` skill would need to:

1. Accept a model name argument instead of assuming statcast-gbm
2. Query the model for its target list and available features
3. Adapt the correlation screening step based on the model's data source
4. Remove statcast-specific SQL expression examples (or make them conditional)

This is mostly a documentation rewrite of `SKILL.md` plus conditional logic for model-specific screening tools.

## What's not worth generalizing

- **`extract_targets` derived-target logic** (iso = slg - avg, babip formula, etc.) — These are domain-specific formulas tied to baseball statistics. Every model that predicts these targets needs the same derivations. This should stay in a shared utility, not be abstracted per-model.
- **Statcast SQL feature expressions** — The `feature candidate --correlate` tool screens raw statcast pitch data. This is inherently statcast-specific and should remain so. Other data sources would need their own screening tools.
- **The experiment skill's domain knowledge** — The skill's value comes from knowing what statcast columns exist, what kinds of features tend to help, and how to interpret baseball-specific residual patterns. A generic skill would lose this domain expertise. Better to have one skill per model family.

## Summary

| Layer | Generic today? | Effort to generalize |
|---|---|---|
| Domain (Experiment, TargetResult) | Yes | None needed |
| Repository (SqliteExperimentRepo) | Yes | None needed |
| Summary service | Yes | None needed |
| Quick-eval / marginal-value services | No (imports gbm_training) | Medium — extract TrainingBackend protocol |
| CLI commands | No (isinstance gate) | Small — extract Experimentable protocol |
| /experiment skill | No (statcast-gbm docs) | Write per-model-family skill variants |

The bottom half of the stack is already a general-purpose experiment journal. The top half is a statcast-gbm-specific experiment runner. Generalization means bridging the two with protocols so that new tabular models can plug into the same rapid-iteration loop without duplicating the infrastructure.
