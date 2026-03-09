# Evaluation Framework Roadmap

The current valuation evaluation (`fbm valuations evaluate`) has two fundamental problems:

1. **Population problem**: It reports MAE and Spearman ρ over ~883 players, ~700 of whom have $0 predicted and $0 actual. This inflates ρ to ~0.71 and dilutes MAE to ~$3.92. When restricted to fantasy-relevant players (predicted OR actual > $0), ρ drops to ~0.03 and MAE rises to ~$15.

2. **Circular target problem**: Evaluating predicted ZAR$ against actual ZAR$ (computed by running ZAR on end-of-season stats) means we're testing whether the formula agrees with itself given different inputs. If the formula has structural biases (e.g., SV+HLD z-scores anti-predict real value), those biases appear in both predicted and actual, masking the problem. The pitcher diagnosis (2026-03-09) showed that formula variants which dramatically improve WAR correlation (0.17→0.25) show *no improvement* on actual ZAR$ because the target itself is distorted by the same formula.

This roadmap addresses both problems: making the evaluator filter to meaningful populations, adding independent evaluation targets beyond ZAR$, and enabling head-to-head comparison of formula variants.

## Status

| Phase | Status |
|-------|--------|
| 1 — Fantasy-relevant filtering | done (2025-03-09) |
| 2 — Independent evaluation targets | in progress |
| 3 — Batter/pitcher stratification and tail accuracy | not started |
| 4 — Compare command and default-to-relevant | not started |

## Phase 1: Fantasy-relevant filtering for valuation evaluation

Add population filters to `ValuationEvaluator.evaluate()` and expose them in the CLI.

### Context

`ValuationEvaluator.evaluate()` accepts no filtering parameters. It matches all predicted valuations against actuals and computes MAE/ρ over the entire set. The projection evaluator (`ProjectionEvaluator`) already supports `top`, `min_pa`, `min_ip` — the valuation evaluator should match this.

The most impactful filter is "fantasy-relevant": players where predicted OR actual value > $0. This cuts the population from ~883 to ~230. A simpler proxy is "top N by predicted rank" (e.g., top 300), which doesn't require actuals to define the filter.

### Steps

1. Add optional parameters to `ValuationEvaluator.evaluate()`: `top: int | None` (top N by predicted rank), `min_value: float | None` (minimum predicted or actual value to include). Keep the current behavior as the default (no filtering).
2. Apply filters after matching predicted vs actual but before computing MAE/ρ. Ensure `n` in the result reflects the filtered count.
3. Add `--min-value` CLI option to `fbm valuations evaluate`. The existing `--top` option currently only controls the mispricings display — repurpose or add a separate `--top-n` for population filtering (distinct from the mispricings `--top`).
4. Update `print_valuation_eval_result()` to indicate when a filter is active (e.g., "season 2024 (231 of 883 matched players, pred|act > $0)").
5. Add tests: verify that filtering reduces `n`, that MAE/ρ change, and that edge cases (empty filter, all filtered out) are handled.

### Acceptance criteria

- `fbm valuations evaluate --season 2024 --system zar --version holdout --league h2h --min-value 0.01` reports n≈231, MAE≈15, ρ≈0.03.
- `fbm valuations evaluate` without filters produces the same output as today (backward compatible).
- `ValuationEvalResult.n` reflects the filtered population.

---

## Phase 2: Independent evaluation targets

Add evaluation against actual WAR and actual H2H category production, not just actual ZAR$.

### Context

Evaluating predicted ZAR$ against actual ZAR$ is circular — the actual values are computed by the same formula with different inputs. When the pitcher diagnosis showed that removing SV+HLD from the predicted formula improved WAR correlation from 0.17 to 0.25, the actual-ZAR$ metric showed *no improvement* because the actual ZAR$ target also overweights SV+HLD.

To validate formula changes, we need targets independent of the formula:
- **Actual WAR**: FanGraphs WAR is an independent measure of real-season value. It has limitations (doesn't map 1:1 to fantasy value) but breaks the circularity.
- **Actual H2H category composite**: Sum each player's actual end-of-season stats weighted by their marginal H2H standings impact. This approximates "how much did this player actually help an H2H team?" without going through the z-score machinery.
- **Top-N hit rate**: What fraction of predicted top-N actually finish in actual top-N? Simple, interpretable, and independent of dollar values.

### Steps

1. Extend `ValuationEvalResult` (or create a parallel `ValuationEvalReport`) to hold multiple target metrics: ρ and MAE vs actual ZAR$, ρ vs actual WAR, top-N hit rate for N=25,50,100.
2. Add actual WAR lookup to the evaluator. Join predicted valuations against batting/pitching stats for the season. Compute Spearman ρ of predicted dollars vs actual WAR for the filtered population.
3. Implement top-N hit rate: for each N in (25, 50, 100), compute the overlap between predicted top-N and actual top-N (by ZAR$ rank). Report as a percentage.
4. Add `--targets war,hit-rate` CLI option to select which independent targets to include (default: all).
5. Update output formatting to display a multi-target summary table.
6. Add tests: verify WAR correlation is computed correctly, hit rate matches manual calculation, missing WAR data is handled gracefully.

### Acceptance criteria

- `fbm valuations evaluate --season 2024 --system zar --version holdout --league h2h --min-value 0.01` reports ρ vs actual ZAR$, ρ vs actual WAR, and top-N hit rates in a single output.
- WAR correlation for pitchers is reported separately and matches notebook findings (ρ ≈ 0.15 for baseline).
- Formula variants that improve WAR correlation are visible in the output even when ZAR$ correlation doesn't change.

---

## Phase 3: Batter/pitcher stratification and tail accuracy

Add position-type breakdown and tail accuracy to the valuation evaluator.

### Context

ZAR ranks batters and pitchers in a single combined pool. The pitcher diagnosis showed that overall ρ ≈ 0.03 masks very different behavior: batters ρ ≈ 0.13-0.18, pitchers ρ ≈ -0.04 to -0.07 (against actual ZAR$). Without stratification, improvements to one pool can be hidden by the other.

Tail accuracy (how well the system ranks the top 25 or 50 players) matters most for auction drafts where the top tier commands premium prices. The tier analysis showed ZAR's top-10 pitcher rankings are inverted (ρ = -0.63 in 2024).

### Steps

1. Add a `stratify: str | None` parameter to `ValuationEvaluator.evaluate()` supporting `"player_type"` (batter vs pitcher split). Return per-cohort results.
2. Implement tail accuracy: compute all metrics restricted to the top-N players by predicted rank (e.g., top 25, top 50). Add a `tail_ns: tuple[int, ...] | None` parameter.
3. Add `--stratify` and `--tail` CLI options to `fbm valuations evaluate`.
4. Update output formatting to display per-cohort metrics and tail accuracy tables. Include independent targets from phase 2 in each cohort.
5. Add tests for stratified evaluation and tail accuracy.

### Acceptance criteria

- `fbm valuations evaluate --stratify player_type` reports separate metrics for batters and pitchers.
- `fbm valuations evaluate --tail` reports metrics for top 25 and top 50 players by predicted rank.
- Stratification and tail compose with phase 1 filters and phase 2 targets.

---

## Phase 4: Compare command and default-to-relevant

Make fantasy-relevant filtering the default and add a head-to-head comparison command for valuation systems/variants.

### Context

After phases 1-3, all the metrics exist but must be explicitly requested, and comparing two systems requires running evaluate twice and eyeballing numbers. A dedicated compare command — analogous to `fbm compare` for projections — is needed for the ongoing valuation-accuracy and pitcher-formula work.

### Steps

1. Change the default behavior of `fbm valuations evaluate` to filter to fantasy-relevant players (predicted OR actual > $0). Add a `--full` flag to opt into the old behavior.
2. Create `fbm valuations compare` CLI command that accepts multiple `system/version` pairs and produces a side-by-side comparison table with ΔMAE, Δρ (vs ZAR$), Δρ (vs WAR), and Δhit-rate.
3. Add a `--check` flag to `fbm valuations compare` that exits non-zero if the second system regresses on the *independent* targets (WAR ρ drop > 0.01 or hit-rate drop > 5%), enabling automated go/no-go gates that aren't circular.
4. Update CLAUDE.md implementation discipline to reference `fbm valuations compare --check` for valuation system validation.
5. Update roadmap documentation references that cite full-population metrics.

### Acceptance criteria

- `fbm valuations evaluate` defaults to fantasy-relevant population; `--full` restores old behavior.
- `fbm valuations compare zar/holdout zar-injury-risk/holdout --season 2024 --league h2h` displays a side-by-side comparison including WAR ρ and hit rates.
- `fbm valuations compare ... --check` gates on independent targets, not circular ZAR$ metrics.

## Ordering

- **Phase 1** has no dependencies and delivers immediate value — correct population filtering for all ongoing work.
- **Phase 2** can run in parallel with phase 1 (independent targets are orthogonal to filtering) but is listed second because filtering is simpler to implement. This phase is the most important for the pitcher formula work — without it, we can't validate formula changes.
- **Phase 3** depends on phases 1-2 (stratification should compose with filters and report all targets).
- **Phase 4** depends on phases 1-2 (compare command needs both filters and independent targets). It should land after existing roadmaps that reference full-population numbers have been updated.
