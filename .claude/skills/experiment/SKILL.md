---
name: experiment
description: Run an autonomous feature-engineering experiment loop on the statcast-gbm model (default). Analyzes residuals, generates feature hypotheses, screens via correlation, tests with fast feedback tools, logs results to the experiment journal, and validates winners. Use when the user asks to "experiment on", "explore features for", or "improve" the statcast-gbm model, or says "experiment on batter/pitcher" without specifying a model. Do NOT use for breakout-bust or playing-time — those have dedicated skills.
allowed-tools: Bash(uv run fbm *)
argument-hint: <player-type> [target] [--budget N]
---

# Statcast GBM Experiment Skill

Run a structured experiment loop to discover features that improve the statcast-gbm model. This skill orchestrates the full pipeline: diagnose weaknesses, generate hypotheses, test candidates, log results, and validate winners.

## Argument parsing

Parse `$ARGUMENTS` for:
- **player-type** (required): `batter` or `pitcher`
- **target** (optional): a specific target stat to focus on (e.g., `slg`, `era`). If omitted, work across all targets for the player type.
- **--budget N** (optional): maximum number of experiment iterations (default: 10)

## Context

**Model:** `statcast-gbm` (HistGradientBoosting, per-target regressors)

**Batter targets:** avg, obp, slg, woba, iso, babip
**Pitcher targets:** era, fip, k_per_9, bb_per_9, hr_per_9, babip, whip

**Training seasons:** Use 2019-2024 (excluding 2020). Holdout is always the last season specified.

**Available statcast columns for aggregation:**
release_speed, release_spin_rate, pfx_x, pfx_z, plate_x, plate_z, launch_speed, launch_angle, hit_distance_sc, barrel, estimated_ba_using_speedangle, estimated_woba_using_speedangle, estimated_slg_using_speedangle, hc_x, hc_y, release_extension

**Raw statcast pitch table columns available for SQL expressions:**
All of the above plus: zone, stand, p_throws, type, events, description, spin_dir, spin_axis, game_date, batter_id, pitcher_id, and more. Use `feature candidate --correlate` to test arbitrary SQL aggregations.

## Experiment loop

Execute this loop up to the budget limit. Each iteration should be purposeful — use prior results to guide the next hypothesis.

### Step 0 — Review prior work

Before generating new hypotheses, check what has already been tried:

```
uv run fbm experiment summary --model statcast-gbm --player-type <type>
uv run fbm experiment search --model statcast-gbm --feature <any-feature-you-plan-to-test>
```

Do NOT re-test features that have already been explored unless you have a materially different hypothesis (e.g., different interaction, different binning).

### Step 1 — Diagnose weaknesses

Identify where the model struggles most:

```
# Worst misses on a specific target
uv run fbm residuals worst-misses statcast-gbm/latest --season 2024 --player-type <type> --target <target> --top 20

# Cohort bias — find systematic over/under-prediction
uv run fbm residuals cohort statcast-gbm/latest --season 2024 --player-type <type> --target <target> --all-dimensions

# Feature distribution gaps between good and bad predictions
uv run fbm residuals gaps statcast-gbm/latest --season 2024 --player-type <type> --target <target> --include-raw
```

Analyze the output to identify patterns:
- Are there cohorts with large, significant bias? (e.g., young batters consistently over-predicted)
- Do worst misses share characteristics not captured by existing features?
- Do feature gaps highlight raw statcast columns not currently in the model?

### Step 2 — Generate hypotheses

Based on the diagnosis, formulate specific, testable hypotheses. Good hypotheses:
- Address a specific weakness found in Step 1
- Have a plausible causal mechanism (e.g., "barrel rate on breaking balls predicts ISO better than overall barrel rate because it captures pitch-type selectivity")
- Can be expressed as a SQL aggregation on the statcast pitch table

Examples of feature ideas:
- Conditional averages: `AVG(launch_speed) FILTER (WHERE barrel = 1)` — barrel exit velocity
- Rate stats: `CAST(SUM(CASE WHEN zone BETWEEN 1 AND 9 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)` — zone rate
- Pitch-type splits: `AVG(release_speed) FILTER (WHERE pitch_type = 'FF')` — fastball velocity
- Interactions: combine two existing features via `fbm feature interact`
- Count-based: `AVG(launch_speed) FILTER (WHERE strikes = 2)` — two-strike exit velo

### Step 3 — Screen candidates with correlation

Before burning compute on model training, check if the candidate has signal:

```
# Test a SQL expression for target correlation
uv run fbm feature candidate "AVG(launch_speed) FILTER (WHERE barrel = 1)" \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --correlate --name barrel_exit_velo

# Check temporal stability
uv run fbm profile stability "AVG(launch_speed) FILTER (WHERE barrel = 1)" \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --all-targets --exclude-season 2020
```

**Gate:** Only proceed to Step 4 if:
- At least one target has |Pearson r| > 0.15 or |Spearman rho| > 0.15
- Stability classification is "stable" or "moderate" for the relevant target(s)

If the candidate fails this gate, log it as a negative result (Step 5) and try the next hypothesis.

### Step 4 — Test with fast feedback and auto-log

Run the candidate through marginal-value with `--experiment` to auto-log results:

```
# Single candidate — auto-logs hypothesis, feature diff, and per-target results
uv run fbm marginal-value statcast-gbm --candidate <column_or_name> \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --experiment "<hypothesis — what you expect and why>" \
  --tags "experiment-skill,<category>"

# Multiple candidates — each gets its own experiment journal entry
uv run fbm marginal-value statcast-gbm --candidate cand1 --candidate cand2 \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --experiment "<hypothesis>" --tags "experiment-skill,<category>"
```

The `--experiment` flag auto-logs: timestamp, hypothesis, model, player type, feature diff (added candidate), train/holdout seasons, per-target RMSE deltas, and a computed conclusion. No separate logging step needed.

Use `--parent-id <N>` to chain related experiments (e.g., variations of the same idea).

Use descriptive tags: `batted-ball`, `plate-discipline`, `pitch-mix`, `interaction`, `binned`, `conditional-avg`, etc.

For single-target quick checks, use `quick-eval` with auto-logging (requires `--baseline`):

```
uv run fbm quick-eval statcast-gbm --target <target> --inject <candidate> \
  --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --baseline <baseline_rmse> \
  --experiment "<hypothesis>" --tags "experiment-skill,<category>"
```

**Interpret results:**
- `avg_delta_pct` < 0 means improvement (lower RMSE)
- Look at per-target deltas — improvement on the target of interest matters most
- A feature that helps one target but hurts many others is not worth pursuing

### Step 5 — Iterate

After each experiment:
1. Review the result — did it match the hypothesis?
2. If the feature helped, consider:
   - Interactions with other features (`fbm feature interact`)
   - Binned versions (`fbm feature bin`)
   - Variations (different filters, different aggregations)
3. If the feature didn't help, consider:
   - Why not? Was the signal absorbed by existing features?
   - Does the residual analysis suggest a different angle?
4. Update the diagnosis if needed — re-run residuals with new context

### Step 6 — Validate winners

After completing the experiment budget, if any candidates showed consistent improvement (negative `avg_delta_pct` across multiple targets):

```
# Compare full feature set A (default) vs B (default + winners)
uv run fbm compare-features statcast-gbm \
  --set-a default \
  --set-b "<default columns>,<winner1>,<winner2>" \
  --player-type <type> \
  --season 2019 --season 2021 --season 2022 --season 2023 --season 2024
```

If the combined set improves across targets, checkpoint it:

```
uv run fbm experiment checkpoint save "<descriptive-name>" \
  --from-experiment <best_experiment_id> \
  --model statcast-gbm \
  --player-type <type> \
  --notes "<summary of what improved and by how much>"
```

## Summary report

After all iterations, present a summary to the user:

1. **Experiments run:** N total (X positive, Y negative, Z screened out at correlation gate)
2. **Best candidates:** List features that showed improvement, with delta_pct per target
3. **Worst ideas:** List features that were expected to help but didn't (and why)
4. **Patterns observed:** What types of features tend to help this model?
5. **Recommended next steps:** Suggestions for further exploration based on patterns found
6. **Checkpoint:** Name of saved checkpoint if a winning feature set was identified

## Rules

- **Never skip logging.** Every experiment gets logged, even failures. Negative results prevent duplicate work.
- **Never skip the correlation gate.** Training is expensive relative to correlation checks. Always screen first.
- **Respect the budget.** Stop at N iterations even if you have more ideas. Present remaining ideas as recommendations.
- **Don't re-test known features.** Always check the experiment journal first.
- **Be specific in hypotheses.** "Maybe exit velocity helps" is not a hypothesis. "Barrel exit velocity (exit velo on barrels only) should predict ISO because barrel quality distinguishes power hitters" is.
- **Report honestly.** If nothing improved, say so. Don't cherry-pick one target's improvement while ignoring regressions on others.
- **Use 2020 exclusion.** Always exclude the 2020 shortened season from training and evaluation seasons.
