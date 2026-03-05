# Playing Time Flexibility Roadmap

The playing time pipeline is currently hardcoded to two external systems (Steamer and ZiPS) for consensus PA/IP. This roadmap generalizes the entire PT pipeline so that PA/IP projections from any source — ATC, FanGraphs Depth Charts, manual overrides, or future systems — can be used both as features in the playing-time model and as direct PT sources in the ensemble, Marcel, and ZAR pipelines.

The end state: a single `playing_time` config parameter (e.g., `playing_time=atc` or `playing_time=consensus:steamer,zips,atc`) selects which system's PA/IP flows through the entire downstream pipeline, and the playing-time model's ridge regression can ingest PA/IP from any registered projection system as features.

## Status

| Phase | Status |
|-------|--------|
| 1 — Configurable consensus systems | done (2026-03-04) |
| 2 — Any-system PT mode in ensemble and Marcel | not started |
| 3 — Generalize PT model features | not started |
| 4 — CLI and validation | not started |

## Phase 1: Configurable consensus systems

Replace the hardcoded Steamer+ZiPS consensus with a configurable list of systems and optional weights.

### Context

`consensus_pt.py` contains `make_consensus_transform()` which computes `(steamer_pa + zips_pa) / 2` with a fallback to whichever is available. This is used by the playing-time model as a feature and by the ensemble/Marcel as a PT mode. Adding a third system (e.g., ATC) requires editing multiple hardcoded locations. The consensus transform, the feature builders, and the ensemble's `build_consensus_lookup()` all assume exactly two systems.

### Steps

1. Refactor `make_consensus_transform()` in `consensus_pt.py` to accept a list of `(system_name, weight)` pairs instead of two hardcoded column names. Default to `[("steamer", 1.0), ("zips", 1.0)]` for backward compatibility. Handle missing systems gracefully (re-normalize weights over available systems per player).
2. Update `build_consensus_lookup()` in `ensemble/engine.py` to accept an arbitrary list of projection sets (not just steamer + zips). It should compute weighted consensus from whatever systems are provided.
3. Update the feature builders in `playing_time/features.py` (`build_batting_pt_features`, `build_pitching_pt_features`) to generate projection-lag features for a configurable list of systems rather than hardcoded `steamer_pa`/`zips_pa`/`steamer_ip`/`zips_ip`.
4. Ensure the derived `consensus_pa`/`consensus_ip` columns still work as before when using the default system list, so existing trained models remain valid.
5. Add tests: consensus with 3 systems, unequal weights, missing systems for some players, single-system fallback.

### Acceptance criteria

- `make_consensus_transform()` accepts N systems with weights; default behavior unchanged.
- `build_consensus_lookup()` works with any number of projection system inputs.
- PT feature builders parameterize which systems contribute to consensus.
- All existing playing-time and ensemble tests pass without modification.
- New tests cover 3-system consensus, weighted consensus, and partial-availability scenarios.

## Phase 2: Any-system PT mode in ensemble and Marcel

Extend the ensemble and Marcel `playing_time` parameter to accept any projection system name, not just `"native"`, `"consensus"`, and `"playing-time-model"`.

### Context

The ensemble's `pt_mode` only understands `"native"` and `"consensus"`. Marcel adds `"playing-time-model"`. If a user wants to use Steamer's PA directly (not averaged with ZiPS), or ATC's PA, there's no way to express that. The system should accept any system name registered in `ProjectionRepo` and look up that system's PA/IP per player.

### Steps

1. Define a PT resolution protocol: given a `playing_time` parameter value, resolve it to a `dict[int, float]` mapping player_id → PA or IP. The resolver should handle:
   - `"native"` — no override, use component systems' own PT (ensemble) or internal formula (Marcel).
   - `"consensus"` — weighted average of configured systems (from phase 1).
   - `"consensus:sys1,sys2,sys3"` — inline consensus specification with named systems.
   - `"<system_name>"` (e.g., `"steamer"`, `"atc"`, `"playing-time-model"`) — look up that system's projections in `ProjectionRepo` and extract PA/IP.
2. Extract the resolver into a shared module (e.g., `domain/pt_resolution.py`) so both ensemble and Marcel use the same logic.
3. Update the ensemble model to use the resolver. Remove the hardcoded `if pt_mode == "consensus"` branch that fetches only steamer+zips projections.
4. Update the Marcel model to use the same resolver, replacing the `_PT_COL` dict and the per-mode feature-building branches.
5. Add tests: resolve PT from a custom system, inline consensus syntax, unknown system raises clear error, resolver returns empty dict for system with no projections.

### Acceptance criteria

- `playing_time="steamer"` in ensemble uses Steamer's PA/IP directly (no averaging).
- `playing_time="consensus:steamer,zips,atc"` computes weighted average of three systems.
- `playing_time="playing-time-model"` still works in both ensemble and Marcel.
- Ensemble and Marcel share the same resolution logic.
- Unknown system name produces a clear error message.
- All existing ensemble and Marcel tests pass.

## Phase 3: Generalize PT model features

Allow the playing-time ridge regression model itself to ingest PA/IP from any set of projection systems as input features, not just Steamer and ZiPS.

### Context

The playing-time model uses `steamer_pa`, `zips_pa` (and IP equivalents) as features in its ridge regression, plus a derived `consensus_pa`/`consensus_ip`. These are hardcoded in `playing_time/features.py` and excluded from the regression coefficient set in specific ways. When a new system is added, the model should be able to use its PA/IP as an additional feature — the ridge regression can then learn how much to trust each source.

### Steps

1. Add a `pt_systems` parameter to the playing-time model config (default `["steamer", "zips"]`). This controls which systems' PA/IP projections are included as features.
2. Update `build_batting_pt_features()` and `build_pitching_pt_features()` to dynamically generate projection-lag features for each system in `pt_systems`.
3. Update the feature exclusion list (features excluded from regression) to dynamically exclude per-system raw columns while keeping the consensus column.
4. Update the ablation groups to handle variable numbers of PT system features.
5. Retrain the playing-time model with the default 2-system config and verify predictions are unchanged (coefficients may differ slightly due to feature ordering, but MAE should be equivalent).
6. Add tests: train with 3 systems, train with 1 system, train with no external systems (pure autoregressive).

### Acceptance criteria

- `pt_systems` config parameter controls which projection systems feed into the ridge regression.
- Default `["steamer", "zips"]` produces equivalent predictions to current model.
- Model can train and predict with 1, 2, or 3+ external systems.
- Model can train with no external systems (autoregressive mode using only historical PA/IP and age).
- Ablation study works correctly with variable system counts.

## Phase 4: CLI and validation

Surface the new flexibility in the CLI and add guardrails so misconfiguration is caught early.

### Context

The new PT options need to be discoverable and validated. Users should be able to specify PT systems via CLI params and get clear feedback when a requested system has no projections for the target season.

### Steps

1. Add a `--playing-time` parameter to relevant `fbm predict` subcommands (ensemble, marcel, playing_time) that accepts the full syntax from phase 2 (`native`, `consensus`, `consensus:sys1,sys2`, or a system name).
2. Add a `--pt-systems` parameter to `fbm train playing_time` and `fbm predict playing_time` to specify which systems' PA/IP to use as features.
3. Add validation at predict time: if the requested PT system has no projections for the target season, emit a clear warning listing available systems and their projection counts.
4. Add a `fbm projections pt-sources --season <year>` command that lists all systems with PA/IP projections for a given season, with player counts per system.
5. Document the PT configuration options in the CLI help text.

### Acceptance criteria

- `fbm predict ensemble --playing-time=atc --season 2025` uses ATC PA/IP.
- `fbm predict ensemble --playing-time=consensus:steamer,zips,atc --season 2025` uses 3-system consensus.
- `fbm train playing_time --pt-systems steamer zips atc --season 2025` trains with 3 input systems.
- `fbm projections pt-sources --season 2025` lists available PT systems.
- Missing system produces a warning, not a silent failure.
- All CLI help text documents the new options.

## Ordering

Phases are sequential — each builds on the prior:

1. **Phase 1** is foundational: without configurable consensus, phases 2–3 have nowhere to plug in new systems.
2. **Phase 2** depends on phase 1 for the consensus infrastructure and delivers the highest user-facing value (any-system PT mode).
3. **Phase 3** depends on phase 1 for the dynamic feature builders and is the most architecturally significant change (model retraining).
4. **Phase 4** depends on phases 2–3 for the options it exposes.

No external roadmap dependencies. The projection-blender roadmap (completed) provides the per-stat routing infrastructure that complements this work — once PT is flexible, users can route rate stats from the GBM and PT from any source via the blender.
