# Experiment System Generalization

Unlock the experiment loop — residual analysis, marginal-value testing, correlation screening, auto-logged journal entries — for any tabular model, not just statcast-gbm. Today the fast-feedback tools are gatekept behind `isinstance(_StatcastGBMBase)` checks, the service layer hardcodes `gbm_training` imports, and the `/experiment` skill is written exclusively for statcast columns and targets. This roadmap extracts protocols at each coupling point so that breakout-bust, playing-time, and future models can plug into the same infrastructure without duplicating it.

## Status

| Phase | Status |
|-------|--------|
| 1 — Experimentable protocol | done (2026-03-05) |
| 2 — Training backend abstraction | done (2026-03-05) |
| 3 — Implement Experimentable for breakout-bust and playing-time | done (2026-03-05) |
| 4 — Per-model experiment skills | not started |

## Phase 1: Experimentable protocol

Define the interface a model must expose to participate in feature-engineering experiments, and replace all `isinstance(_StatcastGBMBase)` gates in the CLI with protocol checks.

### Context

Four CLI commands (`marginal_value.py`, `quick_eval.py`, `compare_features.py`, `validate.py`) guard entry with `isinstance(model_instance, _StatcastGBMBase)` and then reach into private attributes (`_batter_columns`, `_pitcher_columns`, `_batter_training_set_builder`, `_pitcher_training_set_builder`) plus module-level constants (`BATTER_TARGETS`, `PITCHER_TARGETS`). This means no other model can use these tools even though the underlying service logic is model-agnostic.

### Steps

1. Create `src/fantasy_baseball_manager/models/experimentable.py` with a `runtime_checkable` `Experimentable` Protocol:
   - `feature_columns(player_type: str) -> list[str]` — current feature set for the given player type.
   - `targets(player_type: str) -> list[str]` — target stat names for the given player type.
   - `training_data(player_type: str, seasons: list[int], assembler: DatasetAssembler) -> dict[int, list[dict[str, Any]]]` — materialized training rows grouped by season.
   - `player_types() -> list[str]` — which player types this model supports (e.g. `["batter", "pitcher"]`).
2. Make `_StatcastGBMBase` satisfy the `Experimentable` protocol by adding the public methods. The private attributes can remain for internal use; the protocol methods delegate to them.
3. In each gatekept CLI command, replace `isinstance(model_instance, _StatcastGBMBase)` with `isinstance(model_instance, Experimentable)`, and replace all accesses to private attributes / target constants with calls to the protocol methods.
4. Update imports: remove `_StatcastGBMBase` and `BATTER_TARGETS`/`PITCHER_TARGETS` imports from CLI commands that no longer need them.
5. Add tests verifying that the CLI commands work when given any `Experimentable`-satisfying fake model (not just statcast-gbm), including proper error messaging when a model does NOT satisfy the protocol.

### Acceptance criteria

- No CLI command imports `_StatcastGBMBase` or statcast-gbm target constants.
- All four CLI commands accept any model satisfying `Experimentable`.
- `_StatcastGBMBase` satisfies `isinstance(..., Experimentable)` at runtime — existing statcast-gbm workflows are unaffected.
- A non-`Experimentable` model gets a clear error message when passed to these commands.
- All existing tests pass without modification (beyond import path updates).

## Phase 2: Training backend abstraction

Extract the training loop out of `services/quick_eval.py` into a protocol so different model families can supply their own fit/predict backend.

### Context

`quick_eval()`, `marginal_value()`, and `compare_feature_sets()` directly import `extract_features`, `extract_targets`, and `fit_models` from `models/gbm_training.py`. These functions train `HistGradientBoostingRegressor` models and handle derived-target computation (iso, hr_per_9, babip formulas). A playing-time model uses OLS regression, and breakout-bust uses classification — neither can use `fit_models` as-is.

### Steps

1. Define a `TrainingBackend` protocol in `models/experimentable.py` (or a new `models/training_backend.py`):
   - `extract_features(rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]`
   - `extract_targets(rows: list[dict[str, Any]], targets: list[str]) -> dict[str, TargetVector]`
   - `fit(X: list[list[float]], targets: dict[str, TargetVector], params: dict[str, Any]) -> FittedModels`
   - Where `FittedModels` is a protocol with `predict(target: str, X: list[list[float]]) -> ndarray`.
2. Implement `GBMTrainingBackend` wrapping the existing `gbm_training.extract_features`, `extract_targets`, and `fit_models`. This is a thin adapter — no logic changes.
3. Add `training_backend()` to the `Experimentable` protocol returning a `TrainingBackend`.
4. Refactor `quick_eval()`, `marginal_value()`, and `compare_feature_sets()` to accept a `TrainingBackend` parameter instead of importing `gbm_training` directly. The private helper `_score_feature_set` similarly takes a backend.
5. Update CLI commands to pass `model_instance.training_backend()` to the service functions.
6. Update `_StatcastGBMBase` to return `GBMTrainingBackend()` from `training_backend()`.
7. Verify all existing tests pass — service-level tests may need a backend parameter injected.

### Acceptance criteria

- `services/quick_eval.py` has zero imports from `models/gbm_training.py`.
- All three service functions accept a `TrainingBackend` parameter.
- `GBMTrainingBackend` wraps the existing gbm_training functions and passes all existing tests.
- A fake `TrainingBackend` can be injected in tests without touching sklearn.

## Phase 3: Implement Experimentable for breakout-bust and playing-time

Wire the two non-GBM models into the experiment infrastructure so they can use marginal-value, quick-eval, compare-features, and auto-log to the journal.

### Context

With the protocol and backend abstraction in place, each model needs to implement `Experimentable` and provide a `TrainingBackend`. The breakout-bust model is classification (log-loss, not RMSE) and the playing-time model is OLS — each requires a backend that fits its training pipeline.

### Steps

1. **Breakout-bust:**
   - Create `ClassificationTrainingBackend` that wraps `HistGradientBoostingClassifier` training and produces probability predictions.
   - Decide on the evaluation metric: the experiment journal uses RMSE deltas, but classification needs log-loss or accuracy. Either (a) extend `TargetResult` to support an optional `metric_name` field, or (b) treat the 3-class probabilities as regression targets (P(breakout), P(bust)) and use RMSE on calibrated probabilities. Choose the approach that requires less disruption to the existing domain model.
   - Implement `Experimentable` on `BreakoutBustModel` — expose feature columns, targets (the label classes or probability targets), and training data materialization.
   - Verify with a round-trip test: run `marginal_value` on breakout-bust with a candidate feature, confirm it logs to the experiment journal.
2. **Playing-time:**
   - Create `OLSTrainingBackend` wrapping the playing-time `fit_playing_time` / `evaluate_holdout` engine functions. The backend extracts features, fits OLS, and returns predictions.
   - Implement `Experimentable` on `PlayingTimeModel` — expose feature columns (`batting_pt_feature_columns`, `pitching_pt_feature_columns`), targets (`["pa"]` / `["ip"]`), and training data.
   - Verify with a round-trip test: run `marginal_value` on playing-time with a candidate feature, confirm it logs correctly.
3. Add integration tests confirming that `uv run fbm marginal-value breakout-bust --candidate <col> --player-type batter --experiment "test"` and the analogous playing-time command work end-to-end.

### Acceptance criteria

- `BreakoutBustModel` and `PlayingTimeModel` both satisfy `isinstance(..., Experimentable)`.
- `marginal-value`, `quick-eval`, and `compare-features` CLI commands work with all three model families.
- Experiment journal entries from breakout-bust and playing-time are correctly tagged with their model name and queryable via `experiment search --model breakout-bust`.
- Existing statcast-gbm experiment workflows are unaffected.

## Phase 4: Per-model experiment skills

Create model-family-specific `/experiment` skill variants so the autonomous experiment loop works for breakout-bust and playing-time with domain-appropriate guidance.

### Context

The current `/experiment` skill hardcodes statcast-gbm targets, available statcast columns, SQL expression examples, training season strategy, and correlation screening commands. A generic skill would lose domain expertise. Better to have focused variants that know each model's data sources, targets, and feature engineering patterns.

### Steps

1. Refactor the existing `.claude/skills/experiment/SKILL.md` to accept a `--model` argument (or split into per-model files). The loop structure (diagnose → hypothesize → screen → test → log → iterate) is shared; the domain context (targets, column examples, screening tools, season strategy) is model-specific.
2. Create `.claude/skills/experiment-breakout-bust/SKILL.md`:
   - Targets: P(breakout), P(bust) (or the metric chosen in phase 3).
   - Available features: preseason weighted/averaged columns plus ADP features.
   - Screening: correlation of candidates against breakout/bust labels.
   - Season strategy: same 2019-2024 excluding 2020, or adjusted if the label source requires different years.
   - Hypothesis examples: "ADP volatility predicts bust probability because over-drafted players face regression."
3. Create `.claude/skills/experiment-playing-time/SKILL.md`:
   - Targets: pa (batters), ip (pitchers).
   - Available features: fangraphs-sourced playing-time features, lag columns.
   - Screening: correlation of candidates against actual PA/IP.
   - Hypothesis examples: "Spring training PA predicts regular-season PA because it signals lineup position."
4. Update the skill description/trigger in each skill file so Claude selects the right variant based on the user's request (e.g., "experiment on breakout-bust batter" triggers the breakout-bust skill).

### Acceptance criteria

- `/experiment batter` continues to work as before (targets statcast-gbm by default or via explicit model argument).
- `/experiment-breakout-bust batter` runs a full experiment loop with breakout-bust-appropriate targets, columns, and screening.
- `/experiment-playing-time batter` runs a full experiment loop with playing-time-appropriate targets, columns, and screening.
- Each skill variant's domain context (targets, columns, examples) is accurate for its model family.

## Ordering

Phases are strictly sequential: 1 → 2 → 3 → 4.

- **Phase 1** is the minimum viable change — it removes the isinstance gates and makes the CLI commands model-agnostic in their entry checks, though the underlying services still use GBM training. This alone doesn't enable other models, but it's a necessary prerequisite.
- **Phase 2** completes the backend abstraction so the services are truly model-agnostic. After phase 2, any model that implements both protocols can use all fast-feedback tools.
- **Phase 3** is the payoff — it wires breakout-bust and playing-time into the system. This phase depends on both protocols being in place.
- **Phase 4** is independent of the code changes (it's skill documentation) but depends on phase 3 being done so the skills can reference working commands. It can be deferred until the models are actively being experimented on.

Phases 1 and 2 could be combined into a single implementation pass if preferred, since the protocol and backend are tightly related. They're separated here because phase 1 alone has value (cleaner architecture) even without phase 2.
