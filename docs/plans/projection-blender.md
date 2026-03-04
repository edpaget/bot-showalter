# Projection Blender

Extend the existing ensemble model with per-stat routing so that different projection systems can provide different stats in a single blended projection. The goal is to capitalize on the statcast-gbm model's top-300 rate-stat advantage while using Steamer/ZiPS for counting stats that the GBM doesn't produce — yielding a complete projection set that ZAR can turn into draft-day valuations.

The ensemble model (`src/fantasy_baseball_manager/models/ensemble/`) already supports weighted-average blending with per-system version selection and consensus PT normalization. What's missing is the ability to route specific stats (or stat groups) to specific systems, validate that all league-required stats are covered, and wire the output through the valuation pipeline in a single command.

## Status

| Phase | Status |
|-------|--------|
| 1 — Per-stat routing engine | done (2026-03-04) |
| 2 — Stat group presets and config | in progress |
| 3 — Coverage validation and diagnostics | not started |
| 4 — Draft-day pipeline integration | not started |

## Phase 1: Per-stat routing engine

Add a `"routed"` blending mode to the ensemble engine that assigns each output stat to a specific source system (with optional fallback), rather than weight-averaging all systems uniformly.

### Context

The current ensemble has three modes: `weighted_average`, `blend_rates`, and `weighted_spread`. All three apply the same component weights to every stat. This means you can't say "take OBP from statcast-gbm but HR from steamer" — you can only give statcast-gbm a global weight of 0.4 and steamer 0.6, which dilutes the GBM's rate-stat advantage.

The H2H league categories require both counting stats (HR, R, RBI, SB, SO, W, SV+HLD) and rate stats (OBP, ERA, WHIP). statcast-gbm produces rate stats only; Steamer/ZiPS produce both. A routed blend uses each system for what it's best at.

### Steps

1. **Add `routed()` function to `ensemble/engine.py`.** Signature: `routed(system_stats: dict[str, dict[str, Any]], routes: dict[str, str], fallback: str | None) -> dict[str, float]`. For each stat in `routes`, pull the value from the named system's stat_json. If the system doesn't have the stat and `fallback` is set, try the fallback system. Return the merged dict.

2. **Add `"routed"` mode handling in `EnsembleModel.predict()`.** When `mode == "routed"`, read a `routes` dict from `config.model_params` mapping stat names to system names (e.g., `{"obp": "statcast-gbm", "hr": "steamer"}`). Also read an optional `fallback` system name. Pass to the new engine function. The `components` dict still defines which systems to load, but weights are ignored in routed mode (each stat comes from exactly one system).

3. **Support per-stat weighted blending as an alternative.** Add an optional `stat_weights` param: `dict[str, dict[str, float]]` mapping stat → {system → weight}. When present, each stat is weight-averaged using its own per-system weights instead of the global component weights. This is more flexible than pure routing — you can blend OBP 70/30 between statcast-gbm and steamer while taking HR 100% from steamer.

4. **Write tests** covering: basic routing (stat A from system X, stat B from system Y), fallback when primary is missing, per-stat weights, missing stat with no fallback returns empty for that stat, routed mode ignores global component weights.

### Acceptance criteria

- `routed()` engine function exists and is tested independently.
- `EnsembleModel.predict()` accepts `mode = "routed"` with a `routes` dict and produces a merged projection.
- Per-stat `stat_weights` override global `components` weights when provided.
- Fallback system is used when primary system lacks a stat.
- Existing `weighted_average` and `blend_rates` modes are unaffected.

## Phase 2: Stat group presets and config

Add named stat groups so config can route entire groups instead of listing every stat individually. Provide sensible defaults for common league formats.

### Context

Routing individual stats is powerful but verbose. For a 10-category H2H league you'd need 10 entries in `routes`. Most users want to say "rate stats from statcast-gbm, counting stats from steamer" without listing each stat. Stat groups provide this shorthand while still allowing per-stat overrides.

### Steps

1. **Define built-in stat groups as constants** in a new `ensemble/stat_groups.py` module. Groups:
   - `batting_counting`: pa, ab, h, doubles, triples, hr, rbi, r, sb, cs, bb, so, hbp, sf, sh, gdp, ibb
   - `batting_rate`: avg, obp, slg, ops, woba, wrc_plus, iso, babip, k_pct, bb_pct
   - `pitching_counting`: w, l, g, gs, sv, hld, ip, er, so, bb, h, hr
   - `pitching_rate`: era, whip, k_per_9, bb_per_9, fip, xfip, hr_per_9, babip
   - `war`: war

2. **Add `route_groups` param to the ensemble model config.** Format: `dict[str, str]` mapping group name to system name (e.g., `{"batting_rate": "statcast-gbm", "batting_counting": "steamer"}`). The model expands groups into individual stat routes before passing to the engine. Per-stat `routes` entries override group-level assignments.

3. **Support custom group definitions in config.** Allow `stat_groups = {"my_group": ["hr", "rbi", "r"]}` in `model_params` to define ad-hoc groups that can be referenced in `route_groups`.

4. **Add a `"league_required"` pseudo-group** that auto-expands to all stats referenced by the active league's categories (including rate-stat numerators/denominators like `er`, `ip` for ERA). This lets you validate that the blend covers everything the league needs.

5. **Write sample configs** in fbm.toml for common setups: (a) rate-from-gbm / counting-from-steamer, (b) full steamer baseline with GBM rate overrides, (c) three-way blend (steamer + zips + gbm).

6. **Write tests** covering: group expansion, per-stat override of group assignment, custom group definitions, league_required expansion.

### Acceptance criteria

- Built-in stat groups are defined and importable.
- `route_groups` config expands groups to individual stat routes.
- Per-stat `routes` override group-level assignments.
- Custom `stat_groups` definitions work in config.
- At least two sample configs exist in fbm.toml (commented out as examples).

## Phase 3: Coverage validation and diagnostics

Add validation that warns when a blended projection is missing stats required by the league, and a diagnostic command that shows the coverage matrix.

### Context

When routing stats across multiple systems, it's easy to misconfigure and end up with a projection that's missing a required category (e.g., forgetting to route `sv` when the league has a SV+HLD category). This phase adds guardrails.

### Steps

1. **Add coverage validation to `EnsembleModel.predict()`.** After computing routes (from `routes` + `route_groups`), check that every stat in the league's category definitions (both the category key itself and the numerator/denominator components for rate stats) is routed to at least one system. Log warnings for any uncovered stats. If `--check` is passed via config, exit with an error on uncovered stats.

2. **Add a `coverage` diagnostic subcommand** (e.g., `fbm model info ensemble --coverage`). This prints a matrix: rows = stats, columns = systems, cells = whether the system has that stat for the given season. Highlights which system will provide each stat under the current routing config. Marks uncovered league-required stats.

3. **Add a `--dry-run` flag to ensemble predict** that computes and displays the effective routing table (after group expansion and overrides) without actually fetching projections or writing to the database. Useful for verifying config before committing.

4. **Write tests** covering: coverage check passes with valid routing, coverage check fails for missing stat, dry-run produces expected routing table, coverage matrix includes all league categories.

### Acceptance criteria

- Ensemble predict logs warnings for uncovered league-required stats.
- `--dry-run` mode prints the effective routing table without side effects.
- Coverage diagnostic command exists and shows the stat × system matrix.
- Tests verify both pass and fail cases for coverage validation.

## Phase 4: Draft-day pipeline integration

Wire the blended projections through ZAR valuations and verify the end-to-end pipeline produces a usable draft board.

### Context

The blender produces projections; the draft tools consume valuations. This phase confirms the full pipeline works: blend → save → valuate → draft board / tiers / mock draft. It also establishes a recommended default config and benchmarks the blended projections against pure Steamer.

### Steps

1. **Create a default blender config** in fbm.toml for the H2H league. Use statcast-gbm for batting rate stats and pitching rate stats, steamer for all counting stats, consensus PT. Version-pin to `latest` / `2024` as appropriate.

2. **Run the full pipeline**: `fbm predict ensemble`, then `fbm predict zar --param projection_system=ensemble`, then `fbm draft-board export`. Verify the draft board has complete data (no missing categories, reasonable dollar values).

3. **Compare blended valuations against pure-Steamer valuations.** Run ZAR on both and diff the top-300 player rankings. Check that the blended valuations shift players with strong statcast profiles (e.g., high-barrel batters) upward relative to Steamer-only valuations.

4. **Run accuracy comparison** on a holdout season: blend → valuate → compare against actuals. Use `fbm compare ensemble/latest steamer/2024 --season 2024 --stat obp --stat hr --stat era --stat whip --stat so --stat sv` to verify the blend doesn't regress on any league category vs pure Steamer.

5. **Document the recommended workflow** in a short section at the top of this roadmap (or in the README) covering: how to configure the blend, how to run the pipeline, and how to verify results.

### Acceptance criteria

- Default blender config exists in fbm.toml and produces a complete projection set.
- ZAR produces valuations from blended projections without errors.
- Draft board export works end-to-end with blended valuations.
- Blended projections do not regress on any league category vs pure Steamer on at least one holdout season.
- Workflow is documented.

## Ordering

Phases are strictly sequential:

1. **Phase 1** is foundational — the routing engine is required by everything else.
2. **Phase 2** builds on phase 1 by adding config ergonomics (groups, presets).
3. **Phase 3** builds on phase 2 by validating the routing config against league requirements.
4. **Phase 4** is the integration phase that validates the full pipeline end-to-end.

Phase 1 alone is sufficient to produce a working blended projection via CLI params. Phases 2-3 make it easier to configure and harder to misconfigure. Phase 4 confirms real-world value.

No external roadmap dependencies — the ensemble model, ZAR, projection repo, and draft board tools all exist and are functional.
