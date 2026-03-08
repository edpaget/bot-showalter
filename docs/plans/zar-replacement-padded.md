# Replacement-Padded ZAR Roadmap

When a player misses time due to injury, the current `zar-injury-risk` approach (planned in valuation-system-unification) simply scales down counting stats. This undervalues elite injury-prone hitters: a .300 hitter missing 20% of the season isn't worth 80% of his value — he's worth .300 over 80% of PA plus replacement-level production over the remaining 20%. The blended line (.288 over full PA) is more accurate.

This roadmap creates a `zar-replacement-padded` model that fills missed-time PA/IP with position-specific replacement-level production, then runs ZAR on the blended projections. This is a new registered model composing `ZarModel`, following principle 4 ("new valuation approaches are new models, not flags").

Key insight from the Hardball Times research: replacement padding must be applied **globally** because adding replacement-level stats for injury-prone players changes the replacement level for the entire position pool. This requires an iterative or two-pass approach.

## Status

| Phase | Status |
|-------|--------|
| 1 — Replacement-level stat profiles | not started |
| 2 — Projection blending logic | not started |
| 3 — `zar-replacement-padded` model | not started |

## Phase 1: Replacement-level stat profiles

Compute per-position replacement-level stat lines that can be blended into injured players' projections.

### Context

The ZAR engine already computes replacement-level composite z-scores per position, but it doesn't produce replacement-level **stat lines**. To blend replacement production into a player's projection, we need the actual stat rates (AVG, HR rate, etc.) of the replacement-level player at each position. This phase extracts those profiles from the projection pool.

### Steps

1. Create `src/fantasy_baseball_manager/domain/replacement_profile.py` with a `ReplacementProfile` frozen dataclass: `position` (str), `player_type` (str), `stat_line` (dict[str, float]) — the full stat dict of the replacement-level player at this position.
2. Create `src/fantasy_baseball_manager/services/replacement_profiler.py` with a function `compute_replacement_profiles(projections, position_map, roster_spots, num_teams, categories) -> dict[str, ReplacementProfile]`. For each position, identify the replacement-level player (Nth best by composite z, where N = spots x teams) and return their full stat line.
3. Handle edge cases: positions with fewer eligible players than roster spots use the worst eligible player. Positions with no eligible players get a zero-stat profile.
4. Write tests with synthetic projections verifying correct replacement player identification and stat extraction per position.

### Acceptance criteria

- Replacement profiles are computed per position with the correct player's stat line.
- The replacement player is identified consistently with ZAR's existing replacement-level calculation.
- Edge cases (thin positions, no eligible players) are handled gracefully.

## Phase 2: Projection blending logic

Implement the core blending function that mixes a player's projected stats with replacement-level stats based on expected missed time.

### Context

Given a player's projection and their expected days lost, blend their rate and counting stats with the replacement profile for their position. The blend ratio comes from the injury discount factor: `healthy_frac = max(0, 1 - expected_days_lost / 183)`. Rate stats are blended weighted by PA/IP contribution; counting stats are the sum of healthy-fraction originals plus missed-fraction replacement.

### Steps

1. Create `src/fantasy_baseball_manager/services/replacement_padding.py` with `blend_projection(projection, replacement_profile, expected_days_lost) -> Projection`. This function:
   - Computes `healthy_frac = max(0, 1 - expected_days_lost / 183)` and `missed_frac = 1 - healthy_frac`.
   - For counting stats: `blended = original * healthy_frac + replacement * missed_frac` (where replacement counting stats are scaled to the missed PA/IP).
   - For rate stats (AVG, OBP, ERA, WHIP, etc.): weighted average by volume. E.g., blended AVG = `(player_avg * player_pa * healthy_frac + repl_avg * player_pa * missed_frac) / player_pa`.
   - Preserves original PA/IP (the player's slot is filled for the full season).
2. Add `blend_projections(projections, replacement_profiles, injury_map, position_map) -> list[Projection]` that applies blending across the full projection pool.
3. Write tests verifying:
   - Counting stats are correctly blended.
   - Rate stats are weighted by PA/IP fraction.
   - A player with 0 expected days lost gets unchanged projections.
   - A player with 183 expected days lost gets pure replacement-level stats.
   - PA/IP remain at the original (full-season) level.

### Acceptance criteria

- Blended projections preserve full-season PA/IP.
- Rate stats reflect the weighted average of player and replacement performance.
- Counting stats reflect the sum of player and replacement contributions.
- Zero injury risk produces unchanged projections; maximum injury risk produces replacement-level projections.

## Phase 3: `zar-replacement-padded` model

Register the composed model that runs the full replacement-padded pipeline and delegates to ZAR.

### Context

This model composes `ZarModel` (principle 4). It computes replacement profiles from the raw projection pool, blends in replacement production for injured players, then passes the blended projections to ZAR. The two-pass approach is needed: first run ZAR to identify replacement-level players, then use those players' stat lines for blending, then re-run ZAR on the blended projections.

### Steps

1. Create `src/fantasy_baseball_manager/models/zar_replacement_padded/model.py` with `ZarReplacementPaddedModel`, registered as `"zar-replacement-padded"`.
2. Constructor takes the same dependencies as `ZarModel` plus an `InjuryProfiler` protocol dependency (same as the planned `zar-injury-risk` model).
3. `predict()` implementation:
   - Read projections and injury estimates (same as `zar-injury-risk`).
   - **Pass 1:** Run a preliminary ZAR pass on raw projections to identify replacement-level players per position. Use `compute_replacement_profiles()` with the preliminary result.
   - **Pass 2:** Blend projections using `blend_projections()` with the replacement profiles and injury map.
   - **Pass 3:** Run final ZAR on blended projections with `valuation_system="zar-replacement-padded"`.
   - Persist and return results.
4. Wire in `factory.py`: construct `ZarReplacementPaddedModel` when `model_name == "zar-replacement-padded"`, providing `InjuryProfiler` alongside standard ZAR deps.
5. Add `__init__.py` with re-exports and import the module in `models/__init__.py` for auto-registration.
6. Write tests:
   - End-to-end: synthetic projections + injury map → blended valuations with `system="zar-replacement-padded"`.
   - Verify elite injury-prone hitters gain value relative to `zar-injury-risk` (the simple discount approach).
   - Verify healthy players' valuations shift slightly due to changed replacement levels.
   - Verify `fbm valuations lookup <player> --system zar-replacement-padded` works.

### Acceptance criteria

- `fbm predict zar-replacement-padded --param league=<name> --param projection_system=<sys> --season 2026` produces and persists valuations.
- Valuations differ from `zar-injury-risk`: elite injury-prone players are valued higher, marginal players lower.
- The two-pass approach correctly uses replacement-level stat lines from the preliminary ZAR run.
- All existing ZAR tests remain passing.
- Model is discoverable via `fbm models list`.

## Ordering

Phases are strictly sequential: 1 → 2 → 3.

### Dependencies

- **valuation-system-unification phase 1** (planned): That phase makes `ZarModel` system-name-configurable. This roadmap needs the same capability. If valuation-system-unification lands first, this roadmap reuses it. If not, phase 3 here must include the same `valuation_system` parameterization.
- **injury-risk-discount** (done): Provides `InjuryProfiler`, `ExpectedGamesLost`, and the injury estimation pipeline.
