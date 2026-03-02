# Validation Gate

Give an autonomous agent the tools to know *when* a promising result is worth running through the full comparison protocol, and then orchestrate that protocol end-to-end. The full validation (train → predict → compare on 2+ holdout seasons, both top-300 and full pop, with `--check`) is expensive. The agent shouldn't waste time running it on every tweak — but it also shouldn't skip it and declare victory based on quick-eval results alone.

## Status

| Phase | Status |
|-------|--------|
| 1 — Pre-flight confidence estimator | not started |
| 2 — Full validation orchestrator | not started |

## Phase 1: Pre-flight confidence estimator

Based on cross-validated results, estimate the probability that the full `compare --check` will pass before committing to the expensive run.

### Context

The quick-eval and marginal-value tools give fast feedback on single holdout seasons. But the full comparison protocol requires improvements to hold across multiple seasons and both top-300 and full population. A feature that improves SLG by 3% on 2024 holdout might regress on 2023. The pre-flight check uses CV results to estimate consistency: if the improvement is large and consistent across folds, it's likely to pass the full gate.

### Steps

1. Create `src/fantasy_baseball_manager/services/validation_gate.py` with a `preflight_check(cv_results, baseline_cv_results)` function. `cv_results` is a list of per-fold, per-target RMSE values from temporal expanding CV. `baseline_cv_results` is the same for the current model.
2. Compute: per-target win rate across folds (fraction of folds where new < baseline), per-target mean delta, per-target delta standard deviation, and an overall confidence score.
3. The confidence score is: `"high"` if win rate ≥ 0.75 on ≥ 80% of targets, `"medium"` if win rate ≥ 0.6 on ≥ 60% of targets, `"low"` otherwise. These thresholds are configurable.
4. Return a `PreflightResult` frozen dataclass: per-target win rates, per-target mean deltas, overall confidence, and a recommendation (`"proceed"`, `"marginal"`, `"skip"`).
5. Add a `fbm validate preflight <model> --seasons 2021 2022 2023 2024 --candidate-columns col1,col2 --player-type batter` CLI command. It runs CV with the candidate feature set, CV with the baseline feature set, and reports the pre-flight verdict.
6. Write tests with synthetic CV results: one scenario with consistent improvement (expect "proceed"), one with inconsistent results (expect "marginal"), one with regression (expect "skip").

### Acceptance criteria

- Win rate is correctly computed per target across folds.
- Confidence score distinguishes consistent improvement from noise.
- The recommendation correctly maps to the confidence thresholds.
- False positive rate is low — a "proceed" recommendation should rarely fail the full gate.

## Phase 2: Full validation orchestrator

Chain the complete comparison protocol into a single command: train both models, predict, and run `compare --check` on multiple holdout seasons.

### Context

The full validation protocol is currently manual: train old version, train new version, predict for each holdout season, run compare with `--check` for top-300, run compare without `--top` for full population, repeat for a second holdout season. This is error-prone and tedious. The orchestrator automates the entire sequence and reports a single go/no-go verdict.

### Steps

1. Add a `full_validation(model, old_version, new_version, old_params, new_params, holdout_seasons, train_seasons)` function to validation_gate. For each holdout season, it:
   - Trains the old model (if not already trained) on `train_seasons` excluding the holdout.
   - Trains the new model on the same seasons.
   - Generates predictions for both on the holdout season.
   - Runs `compare --check --top 300` (top-300 regression check).
   - Runs `compare --check` without `--top` (full population regression check).
2. Aggregate results across holdout seasons: the new model passes only if `--check` passes on *all* holdout seasons for *both* top-300 and full population.
3. Return a `ValidationResult` frozen dataclass: per-season, per-population pass/fail, per-target RMSE comparisons, and an overall verdict.
4. Add a `fbm validate full <model> --old-version v1 --new-version v2 --holdout 2023 --holdout 2024 --train 2019 2021 2022` CLI command.
5. Support `--new-params '{"feature_columns": [...]}'` to specify the candidate feature set for the new version without having to pre-configure it in `fbm.toml`.
6. If the pre-flight estimator from phase 1 is available, run it first and warn (but don't block) if confidence is "low".
7. Write tests with fake model implementations verifying the orchestration sequence and pass/fail logic.

### Acceptance criteria

- Both models are trained on identical train seasons for each holdout.
- Regression checks run for both top-300 and full population.
- The overall verdict requires all checks to pass across all holdout seasons.
- The orchestrator correctly reuses existing trained models when available.
- Pre-flight warning is shown when confidence is low but does not block the run.
- The full sequence is idempotent — re-running with the same parameters produces the same result.

## Ordering

Phase 1 → Phase 2. The pre-flight estimator is useful standalone — even without the orchestrator, knowing "this probably won't pass" saves time. Phase 2 integrates phase 1 as an optional early warning.

## Dependencies

- **fast-feedback-loop**: The pre-flight estimator uses the same CV infrastructure. Without it, the estimator must re-implement temporal expanding CV internally.
- **experiment-journal**: The orchestrator can optionally log its results as an experiment. Without it, results are only printed to stdout.
