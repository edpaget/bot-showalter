# Direct Rate Stats Roadmap

Both ZAR and SGP currently derive rate stats (AVG, OBP, ERA, WHIP) from counting stat components (e.g. `h/ab` for AVG, `er/ip` for ERA) during their scoring computations. This means that when a routed ensemble supplies superior rate stat projections from statcast-gbm-preseason, both engines ignore them entirely — they recalculate rates from the counting stats, which come from steamer regardless.

This roadmap adds an opt-in mode to both engines that uses projected rate stats directly (e.g. the `avg` field) instead of deriving them from components. This unlocks the ability to pair statcast-gbm-preseason's rate stat accuracy with steamer's counting stat / playing time projections in valuations.

## Status

| Phase | Status |
|-------|--------|
| 1 — Direct rate stat mode in ZAR and SGP engines | in progress |
| 2 — Validation on holdout seasons | not started |

## Phase 1: Direct rate stat mode in ZAR and SGP engines

Add an opt-in `use_direct_rates` parameter to `convert_rate_stats` (ZAR) and `compute_sgp_scores` (SGP), and thread it through both pipelines so it can be activated via model params.

### Context

**ZAR** — `convert_rate_stats` (in `models/zar/engine.py`) handles rate categories by computing `player_rate = resolve_numerator(cat.numerator, stats) / denom` (line 63). For AVG this means `h / ab`. The marginal contribution is then `(player_rate - baseline) * denom`. The baseline is computed as ratio-of-sums across the pool (line 47-49).

**SGP** — `compute_sgp_scores` (in `models/sgp/engine.py`) has the same pattern: line 56 derives rate as `resolve_numerator(cat.numerator, stats) / denom_val` and line 77 does the same per-player. The baseline is the median of derived rates (line 58).

When `use_direct_rates` is enabled, both functions should read the rate stat directly from the stats dict using the category key (e.g. `stats["avg"]`), while still using the denominator field (e.g. `ab`, `pa`, `ip`) for volume weighting (ZAR) or as a zero-check (SGP). Fall back to the derived calculation if the key is missing.

### Steps

1. **ZAR engine** — Add a `use_direct_rates: bool = False` parameter to `convert_rate_stats`. When `True` and the category key exists in the stats dict, use `stats[cat.key]` as the player rate instead of deriving it from `numerator / denominator`. Fall back to the derived calculation if the key is missing (graceful degradation for players without rate stat projections).
2. **ZAR baseline** — Update the baseline computation for direct-rate mode: compute baseline as the weighted mean of direct rates across the pool (weighted by denominator volume), rather than ratio-of-sums. The formula is: `baseline = sum(rate_i * denom_i) / sum(denom_i)` — mathematically equivalent to the current ratio-of-sums when rates are derived, but differs when rates come from an external source.
3. **SGP engine** — Add the same `use_direct_rates: bool = False` parameter to `compute_sgp_scores`. When `True` and the category key exists, use `stats[cat.key]` as the player rate. Update the median baseline to use direct rates when available.
4. **Thread through ZAR pipeline** — `run_zar_pipeline` → `ZarModel._value_pool` → `ZarModel.predict`, accepting from `config.model_params["use_direct_rates"]`.
5. **Thread through SGP pipeline** — `run_sgp_pipeline` → `SgpModel.predict`, accepting from `config.model_params["use_direct_rates"]`.
6. **ZAR tests** — Write tests for `convert_rate_stats` with `use_direct_rates=True`:
   - Rate stat read from stats dict instead of derived from components.
   - Baseline uses volume-weighted mean of direct rates.
   - Missing rate stat key falls back to derived calculation.
   - Counting stats unaffected by the flag.
   - Mixed categories (some rate, some counting) work correctly.
   - Pipeline end-to-end: `run_zar_pipeline` with `use_direct_rates=True` produces different results when stats dicts contain rate fields that disagree with the counting stat components.
7. **SGP tests** — Write analogous tests for `compute_sgp_scores` with `use_direct_rates=True`:
   - Rate stat read from stats dict.
   - Median baseline uses direct rates.
   - Missing key fallback.
   - Pipeline end-to-end: `run_sgp_pipeline` with `use_direct_rates=True`.
8. Verify all existing tests still pass (the default `False` preserves current behavior exactly).

### Acceptance criteria

- `convert_rate_stats(..., use_direct_rates=True)` uses `stats[cat.key]` when present.
- `compute_sgp_scores(..., use_direct_rates=True)` uses `stats[cat.key]` when present.
- ZAR baseline in direct-rate mode is volume-weighted mean of direct rates.
- SGP baseline in direct-rate mode is median of direct rates.
- Both functions fall back to derived rate when key is missing from stats dict.
- Default `use_direct_rates=False` produces identical results to current behavior in both engines.
- `run_zar_pipeline` and `run_sgp_pipeline` accept and pass through `use_direct_rates`.
- `--param use_direct_rates=true` works from the CLI for `zar`, `zar-reformed`, and `sgp`.
- All existing ZAR and SGP engine and model tests pass unchanged.

## Phase 2: Validation on holdout seasons

Generate valuations using a routed ensemble (steamer counting stats + statcast-gbm-preseason rate stats) with `use_direct_rates=True`, and compare accuracy against the steamer-only baseline on holdout seasons.

### Context

The whole point of phase 1 is to let statcast-gbm-preseason's rate stat predictions flow through to valuations. This phase tests whether that actually improves valuation accuracy. The evaluation uses WAR correlation and hit rates as independent targets (not circular ZAR$ metrics).

### Steps

1. Generate routed ensemble projections for 2024 and 2025 (steamer counting + statcast-gbm-preseason rates) if not already present.
2. Run `fbm predict zar-reformed` with `--param projection_system=ensemble --param projection_version=routed-sgbm --param use_direct_rates=true` for 2024 and 2025.
3. Run `fbm predict sgp` with the same ensemble and `use_direct_rates=true` for 2024 and 2025.
4. Run `fbm valuations evaluate` on both seasons for each new version and the steamer baselines.
5. Compare WAR ρ (all, batters, pitchers), hit rates (top-25/50/100), and value MAE.
6. Declare go/no-go: the direct-rate version must not regress WAR ρ by more than 0.01 on either season, and should ideally improve pitcher WAR ρ (where rate stat accuracy matters most).

### Acceptance criteria

- Side-by-side evaluation results documented for both holdout seasons, for both ZAR-reformed and SGP.
- Go/no-go decision recorded in the status table with key metrics.
- If go: update production valuation version to use direct rates with the routed ensemble.
- If no-go: document findings and leave default as `use_direct_rates=False`.

## Ordering

Phase 1 is a pure code change with no data dependencies. Phase 2 depends on phase 1 and on having routed ensemble projections (steamer + statcast-gbm-preseason) in the database for holdout seasons 2024 and 2025 (these already exist from our earlier investigation).
