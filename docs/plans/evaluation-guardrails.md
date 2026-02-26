# Evaluation Guardrails Roadmap

Add decision-support signals to the evaluation and comparison infrastructure so that agents (and humans) can reliably determine whether a model change is an improvement or a regression. The current compare output shows raw per-stat RMSE and R² values in a flat table with no summary, no directional indicators, and no aggregate verdict — making it easy to cherry-pick one improved stat while missing five that degraded. This led to phase 5 of top-300-tuning being reported as a success when the model actually regressed on closer examination.

The fix is two-pronged: enhance the tooling to surface clearer signals, then update the skill file and CLAUDE.md to codify interpretation rules and before/after protocols.

## Status

| Phase | Status |
|-------|--------|
| 1 — Comparison summary and delta signals | not started |
| 2 — Rank-order and tail-accuracy metrics | not started |
| 3 — Regression gate CLI mode | not started |
| 4 — Skill file and CLAUDE.md guardrails | not started |

## Phase 1: Comparison summary and delta signals

Enhance `ComparisonResult` and its output formatting so that a two-system comparison immediately surfaces who won, by how much, and on how many stats.

### Context

Today `print_comparison_result()` renders a flat table of RMSE and R² values. An agent reading this output has to mentally diff every row to determine direction and magnitude. There is no summary line, no delta column, and no win/loss tally. This is the single biggest contributor to misinterpretation — the agent sees numbers but no verdict.

### Steps

1. Add a `ComparisonSummary` domain dataclass capturing per-stat deltas and an aggregate tally. Fields: list of per-stat records (stat name, system-A value, system-B value, absolute delta, relative delta %, winner label) plus aggregate counts (wins, losses, ties for each metric). Build a pure function `summarize_comparison(result: ComparisonResult, baseline_index: int, candidate_index: int) -> ComparisonSummary` in the domain layer.
2. Update `print_comparison_result()`: when exactly two systems are being compared, append a `Δ` column (absolute delta) and a `%Δ` column (relative change from the first system) for both RMSE and R². Color green for improvements, red for regressions. When more than two systems are compared, keep the current format unchanged.
3. Add a summary footer line below the table: `"<candidate> wins N/M stats on RMSE, N/M on R² vs <baseline>"`. Print in bold so it's the first thing an agent reads after the table.
4. Tests: verify `summarize_comparison` correctness with known inputs; verify the summary footer text appears in captured output for two-system comparisons; verify three-system comparisons still render without deltas.

### Acceptance criteria

- Two-system `compare` output includes per-stat delta and %-delta columns for RMSE and R².
- A summary footer line reports win/loss/tie counts for RMSE and R².
- Improvements are green, regressions are red in terminal output.
- Three-or-more-system comparisons render unchanged (no deltas, no footer).
- `ComparisonSummary` is a tested pure-domain object with no CLI dependencies.

## Phase 2: Rank-order and tail-accuracy metrics

Add Spearman rank correlation and top-N tail RMSE to the evaluation toolkit. For fantasy baseball, getting the player ordering right within the top-300 matters more than minimizing absolute error — a model that ranks players correctly but is off by a constant is more useful than one with low RMSE that scrambles rankings.

### Context

Current metrics (RMSE, MAE, Pearson r, R²) all measure absolute prediction accuracy. None of them measure whether the model preserves the correct rank ordering of players within a cohort. A model could improve RMSE slightly (by compressing predictions toward the mean) while making rank order worse — and the current output would report it as an improvement.

Tail accuracy is also missing: how well does the model handle the very best players (top-25, top-50)? These are the most important for fantasy — getting the #1 overall pick wrong costs far more than getting the #250 pick wrong.

### Steps

1. Add `rank_correlation: float` (Spearman) to `StatMetrics`. Compute it in `compute_stat_metrics()` alongside the existing Pearson correlation. Use `scipy.stats.spearmanr` or a pure-Python implementation.
2. Add a `TailAccuracy` dataclass with RMSE computed only on the top-N players (by projected value) for configurable N values (default: 25, 50). Add a `compute_tail_accuracy(comparisons, ns)` function to the domain layer.
3. Update `print_system_metrics()` to include the rank correlation column.
4. Update `print_comparison_result()` to show rank correlation in the two-system delta view (phase 1). Include it in the win/loss tally.
5. Add a `--tail` flag to the `compare` CLI command that appends a tail-accuracy section to the output showing RMSE for the top-25 and top-50 subsets.
6. Tests: verify Spearman calculation on known rankings; verify tail accuracy computes on the correct subset; verify CLI output includes the new columns.

### Acceptance criteria

- `StatMetrics` includes `rank_correlation` (Spearman rho) and it appears in all evaluation output.
- `compare` output includes rank correlation in the delta/summary view.
- `compare --tail` shows RMSE for top-25 and top-50 subsets alongside the full-cohort metrics.
- Rank correlation is included in the phase 1 win/loss tally.

## Phase 3: Regression gate CLI mode

Add a `compare --check` mode that exits with a non-zero status code when the candidate system regresses versus the baseline. This makes it mechanically impossible for an automated workflow to proceed past a regression without explicitly overriding the gate.

### Context

Even with better output formatting and summary lines, an agent can still read a "wins 3/8 stats" summary and decide that's good enough. A hard gate forces acknowledgment. This is analogous to how `pytest` returns non-zero on failure — the tooling enforces the standard, not the operator's judgment.

### Steps

1. Define regression criteria in the domain layer: a candidate fails the check if it loses on a strict majority of stats for RMSE *or* loses on rank correlation for a strict majority of stats (when `--top` is used). Encode this as a `RegressionCheckResult` dataclass with pass/fail, per-stat verdicts, and a human-readable explanation.
2. Add `--check` flag to the `compare` CLI command. When present, the command prints the comparison as usual, then prints the regression check result, and exits with code 1 if the check fails.
3. Require exactly two systems when `--check` is used (first = baseline, second = candidate). Error if more or fewer are provided.
4. Tests: verify exit code 0 when candidate wins majority; verify exit code 1 when candidate loses majority; verify error message when used with != 2 systems.

### Acceptance criteria

- `fbm compare old/v1 new/v2 --season 2025 --check` exits 0 on improvement, 1 on regression.
- Regression is defined as losing a strict majority of stats on RMSE or rank correlation.
- The check result is printed as a clear pass/fail message with per-stat breakdown.
- `--check` with != 2 systems produces a clear error.
- The regression criteria live in the domain layer as a tested pure function.

## Phase 4: Skill file and CLAUDE.md guardrails

Update the `fbm` skill file and `CLAUDE.md` to codify interpretation rules, the before/after comparison protocol, and references to the new tooling from phases 1-3.

### Context

The current skill file's only guidance on interpreting results is: "summarize the results for the user in a clear, readable format." This gives an agent zero framework for distinguishing improvement from regression. CLAUDE.md's implementation discipline section requires verifying acceptance criteria but doesn't mandate before/after comparison for model changes.

These changes should be written *after* the tooling exists so the documentation describes real, available behavior rather than aspirational features.

### Steps

1. Add an "Interpreting evaluation results" section to `.claude/skills/fbm/SKILL.md` covering:
   - **Before/after protocol**: always run `compare old new --season YEAR --top 300 --check` and `compare old new --season YEAR` (full population) before declaring a model change an improvement. Run on at least two holdout seasons.
   - **Reading the summary**: the footer line and delta columns are the primary signal — do not override them with per-stat cherry-picking.
   - **What "better" means**: lower RMSE, higher R², higher rank correlation. A change must win a majority of stats to be considered an improvement. A change that helps one stat but hurts five is a regression.
   - **Common pitfalls**: declaring success from a single stat or single season; confusing "diagnostic showed a problem" with "we fixed it"; not checking full-population for regressions.
2. Add to CLAUDE.md's "Implementation Discipline" section: after any model training or tuning change, run the before/after comparison protocol from the fbm skill. Do not declare improvement unless `--check` passes on all tested seasons for both top-300 and full population.
3. Tests: none (documentation only), but verify the referenced commands actually work by running them manually during implementation.

### Acceptance criteria

- `SKILL.md` has an "Interpreting evaluation results" section with the before/after protocol, reading guidance, and pitfall warnings.
- `CLAUDE.md` implementation discipline section references the `--check` flag and requires before/after comparison for model changes.
- All CLI commands referenced in the documentation are valid and match the actual interface from phases 1-3.

## Ordering

Phases are sequential:

- **Phase 1** is the foundation — the summary/delta output is what all later phases build on and reference.
- **Phase 2** adds rank correlation, which phase 1's summary tally then incorporates. Could technically be done in parallel with phase 1 but the output formatting depends on phase 1's delta view.
- **Phase 3** depends on phases 1-2 because the regression gate uses both RMSE and rank correlation tallies.
- **Phase 4** must be last — it documents the behavior built in phases 1-3.
