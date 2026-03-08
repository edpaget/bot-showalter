# Valuation System Unification Roadmap

Injury-adjusted valuations are currently computed on-the-fly through separate code paths (`--injury-adjusted` flags on `predict` and `rankings`, `fbm report injury-adjusted-values`). This roadmap promotes injury-adjusted valuations to a first-class model registered as `zar-injury-risk`. Running `fbm predict zar-injury-risk` produces valuations persisted under `system="zar-injury-risk"`, queryable via all existing `fbm valuations` commands with `--system zar-injury-risk`.

The new model composes `ZarModel` — it computes injury discounts, injects discounted projections, and delegates to the ZAR pipeline. The only change needed in `ZarModel` is accepting the system name from config instead of hardcoding `"zar"`. After the new model works, legacy `--injury-adjusted` flags and on-the-fly computation paths are removed.

## Status

| Phase | Status |
|-------|--------|
| 1 — Create `zar-injury-risk` model | done (2026-03-08) |
| 2 — Remove legacy injury-adjusted code paths | done (2026-03-08) |

## Phase 1: Create `zar-injury-risk` model

Register a new top-level model that composes `ZarModel` with injury discount logic. The model reads injury history, discounts projections, delegates to ZAR, and persists results under `system="zar-injury-risk"`.

### Context

`ZarModel._value_pool` hardcodes `system="zar"` (model.py:224). To let the composed model persist under a different system name, `ZarModel` needs to read the system name from `config.model_params` instead of hardcoding it. The new `ZarInjuryRiskModel` then sets `valuation_system="zar-injury-risk"` in the config it passes to `ZarModel.predict()`.

### Steps

1. Make `ZarModel` system-name-configurable: read `config.model_params.get("valuation_system", "zar")` in `predict()` and thread it through `_value_pool` instead of hardcoding `"zar"`. Add a test verifying backward compatibility (no param = `system="zar"`).
2. Create `ZarInjuryRiskModel` in a new module `models/zar_injury_risk/model.py`, registered as `"zar-injury-risk"`. Constructor takes the same deps as `ZarModel` plus `InjuryProfiler` (via the existing `InjuryProfiler` service).
3. `ZarInjuryRiskModel.predict()` implementation:
   - Read `seasons_back` from `config.model_params` (default 5).
   - Compute injury estimates via the profiler.
   - Read projections from repo, apply `discount_projections()`.
   - Build a new `ModelConfig` with `valuation_system="zar-injury-risk"` and discounted projections injected.
   - Delegate to `ZarModel.predict()` with this config.
   - Return the `PredictResult`.
4. Wire the model in `factory.py`'s `build_model_context`: when `model_name` is `"zar-injury-risk"`, construct an `InjuryProfiler` and pass it alongside the standard ZAR deps.
5. Add tests:
   - `ZarInjuryRiskModel.predict()` applies injury discounts and produces valuations with `system="zar-injury-risk"`.
   - Discounted counting stats are scaled down; rate stats are preserved.
   - Persisted valuations are queryable via `ValuationRepo.get_by_season(season, system="zar-injury-risk")`.
   - `fbm valuations lookup <player> --system zar-injury-risk` returns the persisted results.

### Acceptance criteria

- `fbm predict zar-injury-risk --param league=<name> --param projection_system=<sys> --season 2026` produces and persists valuations with `system="zar-injury-risk"`.
- `fbm predict zar` still produces `system="zar"` (backward compatible).
- `fbm valuations lookup` and `fbm valuations rankings` work with `--system zar-injury-risk` (reading from DB).
- Injury discount logic is applied: counting stats scaled by `(1 - expected_days_lost / 183)`, rate stats unchanged.
- All existing ZAR tests pass unchanged.

## Phase 2: Remove legacy injury-adjusted code paths

Clean up the on-the-fly injury-adjusted computation paths that are now redundant.

### Context

With `zar-injury-risk` as a persisted model, the on-the-fly computation in `valuations rankings --injury-adjusted`, the `--injury-adjusted` flag on `fbm predict`, and the `injury_valuation.py` service are no longer needed. The `fbm report injury-adjusted-values` command can be simplified to diff two persisted systems from the DB.

### Steps

1. Remove the `--injury-adjusted` flag and associated inline injury discount logic from `fbm predict` (model.py lines 196-228 in the `predict` command).
2. Remove the `--injury-adjusted` flag and on-the-fly computation branch from `fbm valuations rankings`. Remove the `--league`, `--projection-system`, `--projection-version`, `--model`, and `--seasons-back` options that were only needed for on-the-fly computation.
3. Simplify `fbm report injury-adjusted-values` to compare two persisted systems from the DB (e.g., `zar` vs `zar-injury-risk`) instead of recomputing. Keep the `--top` option and output format.
4. Remove or pare down `services/injury_valuation.py` — `compute_injury_adjusted_valuations_list`, `_run_injury_adjusted_predictions`, and `compute_injury_adjusted_deltas` are replaced by DB reads.
5. Remove `InjuryAdjustedValuationsContext` and `build_injury_adjusted_valuations_context` from `factory.py`.
6. Clean up imports in `valuations.py`, `report.py`, `model.py` (CLI), and `factory.py`.
7. Update tests to reflect the new DB-backed flow.

### Acceptance criteria

- `--injury-adjusted` flag is removed from both `fbm predict` and `fbm valuations rankings`.
- `fbm valuations rankings --system zar-injury-risk` works (DB read only).
- `fbm report injury-adjusted-values` diffs two persisted systems from the DB.
- `services/injury_valuation.py` no longer contains on-the-fly computation functions.
- `InjuryAdjustedValuationsContext` is removed from the factory.
- All tests pass.

## Ordering

Phases are strictly sequential:

1. **Phase 1** must land first — the new model must exist and produce persisted valuations before legacy paths can be removed.
2. **Phase 2** depends on phase 1 — safe to remove old code only after the new model is working.

No external roadmap dependencies. The injury-risk-discount roadmap (done) provides all prerequisite infrastructure (profiler, estimator, discount functions).
