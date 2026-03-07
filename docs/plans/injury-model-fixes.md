# Injury Model Fixes

Three issues surfaced when exercising the injury risk reports end-to-end:

1. **Player name lookup is broken for full names.** The injury profiler uses `search_by_name()` (wildcard LIKE against `name_first` / `name_last` individually), so "Bo Bichette" matches nothing because `%Bo Bichette%` doesn't match either column. Meanwhile, other services (projection lookup, valuation lookup) already use `resolve_players()` from `name_utils.py`, which handles "First Last", "Last, First", accents, nicknames, and suffixes. The fix is to provide a single shared name-resolution service that all player-name CLI commands wire through.

2. **Expected days lost estimates are unrealistically low.** Max Scherzer (7 IL stints, 105 days lost in 2024) gets only 2.3 expected days. The root cause: `estimate_games_lost()` only reads `profile.recent_stints` (last 2 seasons), so a player whose entire injury history falls in older seasons appears healthy. Additionally, the `days` field on stints is often `None` (only populated on activations), so the estimator defaults to 15 days per stint regardless of actual IL type (10/15/60-day). And with only 1 season of data loaded, `seasons_tracked=5` but `recent_stints` only covers 2024, making the recency-weighted average far too low.

3. **Only 2024 IL data is loaded.** The `fbm ingest il` command was only run for 2024. The profiler computes `pct_seasons_with_il` as 20% (1 of 5 seasons) and `avg_days_per_season` as 21.0 for a player with 105 days lost — both wildly inaccurate given the missing 2021-2023 data. Multi-year data is essential for the model to work as designed.

## Status

| Phase | Status |
|-------|--------|
| 1 -- Unified player name resolver | in progress |
| 2 -- Multi-year IL data ingestion | not started |
| 3 -- Games-lost estimator accuracy | not started |

## Phase 1: Unified player name resolver

Extract a shared `PlayerNameResolver` service that wraps the existing `resolve_players()` logic from `name_utils.py`, and wire it into every CLI command and service that accepts a player name.

### Context

`resolve_players()` in `name_utils.py` already handles "First Last", "Last, First", single-word last-name queries, accent stripping, nickname aliases, and suffix removal. It uses `search_by_last_name_normalized()` which leverages the SQLite `strip_accents()` custom function. This is strictly better than `search_by_name()` (wildcard LIKE), which fails on full names and returns false positives (e.g., "Rendon" matching "Brendon Davis").

Several services already use `resolve_players()`: `ProjectionLookupService`, `ValuationLookupService`. But `InjuryProfiler`, `PlayerBiographyService`, the keeper CLI commands, and the draft board CLI all still call `search_by_name()` directly.

### Steps

1. Create a `PlayerNameResolver` protocol in `repos/protocols.py` (or `services/`) with a single method `resolve(name: str) -> list[Player]`. Implement it as a thin wrapper around `resolve_players(player_repo, name)`.
2. Refactor `InjuryProfiler` to accept `PlayerNameResolver` (or just use `resolve_players`) instead of calling `self._player_repo.search_by_name()`. Update `lookup_profile()` and `estimate_player_games_lost()`.
3. Refactor `PlayerBiographyService.search()` to use `resolve_players()` instead of `search_by_name()`.
4. Refactor `_resolve_player_id()` in the keeper CLI and `_resolve_roster_names()` in the draft board CLI to use `resolve_players()`.
5. Update the `search_players` LLM tool to use `resolve_players()`.
6. Add integration-style tests verifying that "Bo Bichette", "Bichette, Bo", and "Bichette" all resolve to the same player through the injury profiler and other commands.
7. Consider deprecating or removing `search_by_name()` from the `PlayerRepo` protocol if no callers remain.

### Acceptance criteria

- `fbm report injury-profile "Bo Bichette"` returns Bo Bichette's profile (not Dante Bichette or no result).
- `fbm report injury-estimate "Max Scherzer" --season 2026` returns Scherzer's estimate.
- All CLI commands that accept a player name use the same resolution logic.
- Accent-insensitive matching works: "Acuna" matches "Ronald Acuna Jr."
- Ambiguous single-word queries still work: "Bichette" returns a result (first match or disambiguates).

## Phase 2: Multi-year IL data ingestion

Load IL stint data for 2021-2025 so the injury model has the multi-year history it was designed to use.

### Context

The `fbm ingest il --season <year>` command already supports multi-season ingestion. The problem is purely operational: only 2024 was ever loaded. With 1 season of data, `avg_days_per_season` and `pct_seasons_with_il` are meaningless, and the games-lost estimator's recency weighting has nothing to weight across.

The MLB Stats API (`statsapi.mlb.com/api/v1/transactions`) serves historical transaction data going back many years. The existing ingestion pipeline parses IL placements, activations, and transfers from transaction descriptions.

### Steps

1. Run `fbm ingest il` for seasons 2021, 2022, 2023, and 2025 to populate the `il_stint` table with 5 years of data.
2. Verify data quality: check that each season has a reasonable number of stints (typically 600-1000+ IL placements per season). Log any seasons with unexpectedly low counts.
3. Spot-check a few known injury-prone players (e.g., a pitcher with Tommy John history) to verify their multi-year IL stint history is complete.
4. Add a `fbm data il-coverage` (or similar) command or extend the existing `fbm data status` to show IL stint counts per season, so data gaps are visible.
5. Update the injury-risk-discount roadmap description to note that 5 years of IL data (2021-2025) is the expected baseline.

### Acceptance criteria

- `il_stint` table has data for seasons 2021-2025.
- Each season has a plausible number of stints (no empty seasons).
- `fbm report injury-risks --season 2025 --seasons-back 5` shows meaningful multi-year aggregates (e.g., `pct_seasons_with_il` > 20% for chronically injured players).
- `fbm report injury-profile "Scherzer"` shows stints across multiple seasons, not just 2024.

## Phase 3: Games-lost estimator accuracy

Fix the estimator to produce realistic expected-days-lost values by using all available stint data (not just `recent_stints`) and correctly handling stint duration.

### Context

The current `estimate_games_lost()` has three problems:

1. **Only reads `recent_stints`** (last 2 seasons from `InjuryProfile`). All older history is ignored. A player with 5 IL stints across 2021-2023 but a clean 2024 looks completely healthy.
2. **Defaults to 15 days when `stint.days` is None.** The `days` field is only populated on activation transactions. Placement-only stints (no matching activation) get 15 days regardless of whether they were 10-day or 60-day IL placements. The `il_type` field ("10-day", "15-day", "60-day") is always present and should be used as the fallback.
3. **No floor for high-risk players.** A player with 5+ stints across 5 seasons can still get an estimate of 2-3 days if the stints happened in the "wrong" seasons relative to the recency weights.

The `_compute_days()` helper in `injury_profiler.py` already handles the IL-type fallback correctly (line 35: `_IL_TYPE_DEFAULTS.get(stint.il_type, 15)`), but `estimate_games_lost()` in `games_lost_estimator.py` doesn't use it — it has its own inline `stint.days if stint.days is not None else 15` (line 42).

### Steps

1. **Use all stints, not just `recent_stints`.** Change `estimate_games_lost()` to accept the full list of IL stints (or change `InjuryProfile` to include all stints, not just recent ones). Apply recency weights across all seasons — older seasons still get weight 1, so they contribute but don't dominate.
2. **Use `_compute_days()` for duration.** Extract the days-computation logic from `injury_profiler.py` into a shared utility (or import it) so the estimator uses IL-type defaults (10/15/60 days) when `days` is None, rather than always defaulting to 15.
3. **Add a minimum-risk floor for injury-prone players.** If a player has `total_stints >= N` (e.g., 4+) across the lookback window, set a floor on expected days (e.g., at least `_BASE_RATE_DAYS`) so chronic injury risk isn't averaged away.
4. **Tune base-rate regression.** With multi-year data now available, evaluate whether `_BASE_RATE_DAYS = 12.0` and `_FULL_CREDIBILITY_SEASONS = 6` are well-calibrated. Run the estimator against 2024 actuals using 2019-2023 history and compare predicted vs actual days lost. Adjust constants if needed.
5. **Update tests.** Rewrite the games-lost estimator tests to cover multi-season histories, IL-type-aware duration defaults, and the minimum-risk floor. Add a test case matching Scherzer's profile to verify the estimate is in a realistic range (e.g., 30+ expected days for a player with 7 stints and 105 days in one season).
6. **Re-run reports.** After fixes, re-run `fbm report games-lost --season 2026 --top 20` and `fbm report injury-estimate "Scherzer" --season 2026` to verify estimates are realistic.

### Acceptance criteria

- Max Scherzer's expected days lost is materially higher than baseline (e.g., 25+ days, not 2.3).
- A clean-history player still gets a nonzero baseline estimate (~10-15 days).
- IL type is used for duration when `days` is None: a 60-day IL placement defaults to 60, not 15.
- The estimator uses all available seasons of data, with appropriate recency weighting.
- Tests cover multi-year histories, single-season outliers, and the edge case of all stints being old (3+ seasons ago).

## Ordering

Phase 1 (name resolver) is independent and can land first — it's a pure code quality fix. Phase 2 (data ingestion) should precede phase 3 because the estimator accuracy work needs multi-year data to test against. Phase 3 depends on phase 2 for realistic validation.

```
Phase 1 (name resolver) ──────────────────────────►
Phase 2 (multi-year IL data) ──► Phase 3 (estimator accuracy)
```

## Dependencies

- Depends on the existing `name_utils.py` (`resolve_players`, `normalize_name`) from the player-bio-fuzzy-team roadmap (done).
- Depends on the injury-risk-discount roadmap (done) which built the profiler, estimator, and CLI commands.
- No downstream blockers — these are fixes to an already-shipped feature.
