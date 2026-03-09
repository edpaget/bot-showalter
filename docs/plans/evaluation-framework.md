# Evaluation Framework Roadmap

The current valuation evaluation (`fbm valuations evaluate`) reports MAE and Spearman ρ over the full player population (~883 players for 2024). This is misleading: ~700 of those players have $0 predicted and $0 actual, inflating ρ (~0.71) and diluting MAE (~$3.92). When restricted to fantasy-relevant players (predicted OR actual > $0), ρ drops to ~0.03 and MAE rises to ~$15. This was discovered during the valuation-accuracy phase 2 holdout validation (2026-03-08).

The projection evaluator (`fbm compare`) already supports top-N filtering, min-PA/IP thresholds, stratification by age/experience/top-300, and tail accuracy — none of which exist for valuation evaluation. This roadmap brings the valuation evaluator to parity and makes fantasy-relevant filtering the default, so future go/no-go decisions are based on meaningful populations.

## Status

| Phase | Status |
|-------|--------|
| 1 — Fantasy-relevant filtering for valuation evaluation | not started |
| 2 — Batter/pitcher stratification and tail accuracy | not started |
| 3 — Default-to-relevant and compare command | not started |

## Phase 1: Fantasy-relevant filtering for valuation evaluation

Add population filters to `ValuationEvaluator.evaluate()` and expose them in the CLI, so evaluations can target the players that matter.

### Context

`ValuationEvaluator.evaluate()` accepts no filtering parameters. It matches all predicted valuations against actuals and computes MAE/ρ over the entire set. The projection evaluator (`ProjectionEvaluator`) already supports `top`, `min_pa`, `min_ip` — the valuation evaluator should match this.

The most impactful filter is "fantasy-relevant": players where predicted OR actual value > $0. This cuts the population from ~883 to ~230 and exposes the true signal. A simpler proxy is "top N by predicted rank" (e.g., top 300), which doesn't require actuals to define the filter.

### Steps

1. Add optional parameters to `ValuationEvaluator.evaluate()`: `top: int | None` (top N by predicted rank), `min_value: float | None` (minimum predicted or actual value to include). Keep the current behavior as the default (no filtering).
2. Apply filters after matching predicted vs actual but before computing MAE/ρ. Ensure `n` in the result reflects the filtered count.
3. Add `--min-value` CLI option to `fbm valuations evaluate`. The existing `--top` option currently only controls the mispricings display — repurpose or add a separate `--top-n` for population filtering (distinct from the mispricings `--top`).
4. Update `print_valuation_eval_result()` to indicate when a filter is active (e.g., "season 2024 (231 of 883 matched players, pred|act > $0)").
5. Add tests: verify that filtering reduces `n`, that MAE/ρ change, and that edge cases (empty filter, all filtered out) are handled.

### Acceptance criteria

- `fbm valuations evaluate --season 2024 --system zar --version holdout --league h2h --min-value 0.01` reports n=231, MAE≈14.98, ρ≈0.03 (matching the notebook analysis).
- `fbm valuations evaluate` without filters produces the same output as today (backward compatible).
- `ValuationEvalResult.n` reflects the filtered population.

---

## Phase 2: Batter/pitcher stratification and tail accuracy

Add position-type breakdown and tail accuracy to the valuation evaluator, mirroring capabilities the projection evaluator already has.

### Context

ZAR ranks batters and pitchers in a single combined pool. A system that's great at ranking batters but terrible at ranking pitchers (or vice versa) would show mediocre overall ρ, hiding actionable signal. The projection evaluator supports stratification by age, experience, and top-300 status — the valuation evaluator should support at least a batter/pitcher split.

Tail accuracy (how well the system ranks the top 25 or 50 players) is especially important for auction drafts where the top tier commands premium prices.

### Steps

1. Add a `stratify: str | None` parameter to `ValuationEvaluator.evaluate()` supporting at minimum `"player_type"` (batter vs pitcher split). Return a `dict[str, ValuationEvalResult]` keyed by cohort label, or extend the result type.
2. Implement tail accuracy: compute MAE and ρ restricted to the top-N players by predicted rank (e.g., top 25, top 50). Add a `tail_ns: tuple[int, ...] | None` parameter.
3. Add `--stratify` and `--tail` CLI options to `fbm valuations evaluate`.
4. Update output formatting to display per-cohort metrics and tail accuracy tables.
5. Add tests for stratified evaluation and tail accuracy.

### Acceptance criteria

- `fbm valuations evaluate --stratify player_type` reports separate MAE/ρ for batters and pitchers.
- `fbm valuations evaluate --tail` reports MAE/ρ for top 25 and top 50 players by predicted rank.
- Stratification and tail can be combined with phase 1 filters (e.g., `--min-value 0.01 --stratify player_type`).

---

## Phase 3: Default-to-relevant and valuation compare command

Make fantasy-relevant filtering the default for valuation evaluation and add a head-to-head comparison command for valuation systems.

### Context

After phase 1, the filter exists but must be explicitly requested. Go/no-go decisions in roadmaps like valuation-accuracy should use the relevant population by default, not the full population. The current workflow for comparing two valuation systems (e.g., zar vs zar-injury-risk) requires running `fbm valuations evaluate` twice and eyeballing the numbers. A dedicated compare command — analogous to `fbm compare` for projections — would streamline this.

### Steps

1. Change the default behavior of `fbm valuations evaluate` to filter to fantasy-relevant players (predicted OR actual > $0). Add a `--full` flag to opt into the old behavior.
2. Update all roadmap documentation references that cite full-population metrics to note the population used.
3. Create `fbm valuations compare` CLI command that accepts multiple `system/version` pairs and produces a side-by-side comparison table with ΔMAE and Δρ, similar to `fbm compare` for projections.
4. Add a `--check` flag to `fbm valuations compare` that exits non-zero if the second system regresses vs the first (MAE increase or ρ drop > 0.01), enabling automated go/no-go gates.
5. Update CLAUDE.md implementation discipline to reference `fbm valuations compare --check` for valuation system validation.

### Acceptance criteria

- `fbm valuations evaluate` defaults to fantasy-relevant population; `--full` restores old behavior.
- `fbm valuations compare zar/holdout zar-injury-risk/holdout --season 2024 --league h2h` displays a side-by-side comparison table.
- `fbm valuations compare ... --check` exits 0 on improvement, non-zero on regression.

## Ordering

- **Phase 1** has no dependencies and delivers the most immediate value — correct evaluation numbers for ongoing valuation-accuracy work.
- **Phase 2** depends on phase 1 (filters should compose with stratification). It adds depth but is lower priority than getting the basic filter in place.
- **Phase 3** depends on phase 1 (changes the default). It should land after existing roadmaps that reference full-population numbers have been updated or completed, to avoid confusion during in-flight work.
