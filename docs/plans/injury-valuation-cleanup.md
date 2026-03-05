# Injury Valuation Cleanup Roadmap

Address architectural warnings from the Phase 3 injury-risk-discount review. Three issues were identified: hardcoded `ZarModel` in CLI commands (principle 4), business logic in CLI commands (principle 5), and a models→services layer violation (principle 8). Each phase is independent.

## Status

| Phase | Status |
|-------|--------|
| 1 — Extract injury-adjusted valuation service | not started |
| 2 — Use model registry instead of hardcoded ZarModel | not started |
| 3 — Inject discount function into ZarModel | not started |

## Phase 1: Extract injury-adjusted valuation service

Move the ~80-line orchestration workflows from `report.py` and `valuations.py` into a service function.

### Context

The `report injury-adjusted-values` command and the `valuations rankings --injury-adjusted` path both contain inline orchestration: read original valuations, compute injury discounts, re-run ZAR, join results, compute deltas. This logic should live in a service so it can be reused by the agent or HTTP interaction modes without duplication.

### Steps

1. Create `src/fantasy_baseball_manager/services/injury_valuation.py` with a function like `compute_injury_adjusted_deltas(...)` that accepts repo protocols, a profiler, league config, and season parameters. Returns `list[InjuryValueDelta]`.
2. Extract the shared "discount + re-run ZAR + build PlayerValuation list" logic into a second service function (e.g., `compute_injury_adjusted_valuations(...)`) used by both the rankings and report commands.
3. Slim down both CLI commands to thin adapters: parse args, call service, format output.
4. Re-export new functions from `services/__init__.py`.
5. Tests for the new service functions using in-memory SQLite.

### Acceptance criteria

- `report.py` and `valuations.py` injury-adjusted paths are under 15 lines each (excluding arg parsing).
- New service functions have full test coverage.
- Existing CLI behavior is unchanged.

## Phase 2: Use model registry instead of hardcoded ZarModel

Replace direct `ZarModel` instantiation with registry dispatch in the injury-adjusted valuation commands.

### Context

Both `report.py` and `valuations.py` import and instantiate `ZarModel` directly. Principle 4 requires CLI commands to operate on model names passed as arguments, dispatching through the registry. This becomes important when a second valuation model is added.

### Steps

1. Add a `--model` parameter (default `"zar"`) to `report injury-adjusted-values` and `valuations rankings --injury-adjusted`.
2. Use `ModelRegistry.get(model_name)` to resolve the model class instead of importing `ZarModel` directly.
3. Remove the direct `ZarModel` import from both CLI files.
4. Update the service function (from phase 1) to accept the model via protocol rather than constructing it internally.

### Acceptance criteria

- No direct `ZarModel` import in `report.py` or `valuations.py`.
- `--model zar` produces identical output to the current hardcoded behavior.
- Architecture test `test_no_bypass_imports` passes without new allowlist entries for these files.

## Phase 3: Inject discount function into ZarModel

Eliminate the models→services layer violation by injecting discounted projections rather than importing `discount_projections` in the model layer.

### Context

`models/zar/model.py` imports `discount_projections` from `services.injury_discount`, which violates unidirectional layer dependencies (models should not import services). The current workaround is a known exception in `test_layer_dependencies.py` and a bypass allowlist entry in `test_reexports.py`.

### Steps

1. Change `ZarModel.predict()` to accept pre-discounted projections via `config.model_params["projections"]` (or a similar mechanism), removing the need to call `discount_projections` internally.
2. Move the discount-then-predict orchestration into the service layer (from phase 1) or the CLI composition root.
3. Remove the `from fantasy_baseball_manager.services.injury_discount import discount_projections` import from `model.py`.
4. Remove the known exception from `test_layer_dependencies.py` and the bypass allowlist entry from `test_reexports.py`.

### Acceptance criteria

- `models/zar/model.py` has zero imports from `services/`.
- `KNOWN_EXCEPTIONS` for `models/zar/model.py` is reduced (ideally to just `player_eligibility`, or empty if that can also be resolved).
- All architecture tests pass without new exceptions.
- Existing predictions are unchanged.
