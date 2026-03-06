---
name: experiment-playing-time
description: Run an autonomous feature-engineering experiment loop on the playing-time regression model. Tests candidate features against PA (batters) and IP (pitchers) targets using marginal-value with auto-logging. Use when the user asks to "experiment on playing-time", "improve playing-time", or "explore features for playing time".
allowed-tools: Bash(uv run fbm *)
argument-hint: <player-type> [--budget N]
---

# Playing-Time Experiment Skill

Run a structured experiment loop to discover features that improve the playing-time model. This skill orchestrates the full pipeline: diagnose weaknesses, generate hypotheses, test candidates, log results, and validate winners.

## Argument parsing

Parse `$ARGUMENTS` for:
- **player-type** (required): `batter` or `pitcher`
- **--budget N** (optional): maximum number of experiment iterations (default: 10)

Note: target is implicit — `pa` for batters, `ip` for pitchers.

## Context

**Model:** `playing_time` (OLS/ridge regression)

**Targets:** `pa` (batters), `ip` (pitchers)

**Training seasons:** Use 2019-2024 (excluding 2020). Holdout is always the last season specified.

**Batter features:**
`age`, `pa_1`, `pa_2`, `pa_3`, `war_1`, `war_2`, `consensus_pa`, `pt_trend`, `war_above_2`, `war_above_4`, `war_below_0`, `war_trend`, `age_pt_factor`

**Pitcher features:**
`age`, `ip_1`, `ip_2`, `ip_3`, `g_1`, `g_2`, `g_3`, `gs_1`, `war_1`, `war_2`, `consensus_ip`, `pt_trend`, `war_above_2`, `war_above_4`, `war_below_0`, `war_trend`, `starter_ratio`, `age_pt_factor`

**Data source:** FanGraphs (not statcast pitch table). Features are lag columns from prior seasons, IL stint data, consensus projections, and derived transforms. New candidate features should come from fangraphs tables or be derived transforms of existing features.

**Key difference from statcast-gbm:** Playing-time uses FanGraphs-sourced features, not statcast. There is no `feature candidate --correlate` tool available. OLS training is fast, so skip the correlation screening gate and test directly with marginal-value.

## Experiment loop

Execute this loop up to the budget limit. Each iteration should be purposeful — use prior results to guide the next hypothesis.

### Step 0 — Review prior work

Before generating new hypotheses, check what has already been tried:

```
uv run fbm experiment summary --model playing_time --player-type <type>
uv run fbm experiment search --model playing_time --feature <any-feature-you-plan-to-test>
```

Do NOT re-test features that have already been explored unless you have a materially different hypothesis.

### Step 1 — Diagnose weaknesses

Identify where the model struggles most:

```
# Worst misses on a specific target
uv run fbm residuals worst-misses playing_time/latest --season 2024 --player-type <type> --target <target> --top 20

# Cohort bias — find systematic over/under-prediction
uv run fbm residuals cohort playing_time/latest --season 2024 --player-type <type> --target <target> --all-dimensions

# Feature distribution gaps between good and bad predictions
uv run fbm residuals gaps playing_time/latest --season 2024 --player-type <type> --target <target> --include-raw
```

Where `<target>` is `pa` for batters or `ip` for pitchers.

Analyze the output to identify patterns:
- Are there cohorts with large, significant bias? (e.g., young players or platoon players systematically over/under-predicted)
- Do worst misses share characteristics not captured by existing features? (e.g., players who changed teams, injury-prone players)
- Do feature gaps highlight data dimensions not currently in the model?

### Step 2 — Generate hypotheses

Based on the diagnosis, formulate specific, testable hypotheses. Good hypotheses:
- Address a specific weakness found in Step 1
- Have a plausible causal mechanism linking the feature to playing time outcomes
- Can be derived from fangraphs data or transforms of existing features

Examples of feature ideas:
- **Spring training PA/IP:** Spring training playing time predicts regular-season PA because it signals lineup position and health
- **IL recurrence rate:** Proportion of recent seasons with IL stints predicts reduced playing time because chronic injuries limit availability
- **ADP-derived rank:** ADP rank can proxy for team confidence in playing time — highly drafted players get more opportunities
- **Team depth:** Number of roster competitors at the same position affects playing time allocation
- **Age-squared:** Quadratic age term to capture non-linear aging effects on playing time
- **Consistency:** Standard deviation of PA/IP across prior seasons — volatile playing time may predict future volatility
- **WAR-PA interaction:** `war_1 * pa_1` — high-WAR players with high PA are most likely to maintain playing time
- **Rookie indicator:** First-year players may have different playing-time dynamics than veterans
- **Games started ratio trends:** Change in `starter_ratio` year-over-year for pitchers signals role changes

### Step 3 — Test with fast feedback and auto-log

Since OLS training is fast, skip the correlation screening gate and test directly with marginal-value:

```
# Single candidate — auto-logs hypothesis, feature diff, and per-target results
uv run fbm marginal-value playing_time --candidate <column_or_name> \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --experiment "<hypothesis — what you expect and why>" \
  --tags "experiment-skill,<category>"

# Multiple candidates — each gets its own experiment journal entry
uv run fbm marginal-value playing_time --candidate cand1 --candidate cand2 \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --experiment "<hypothesis>" --tags "experiment-skill,<category>"
```

The `--experiment` flag auto-logs: timestamp, hypothesis, model, player type, feature diff (added candidate), train/holdout seasons, per-target RMSE deltas, and a computed conclusion.

Use `--parent-id <N>` to chain related experiments (e.g., variations of the same idea).

Use descriptive tags: `injury`, `depth`, `age-curve`, `interaction`, `trend`, `role-change`, `rookie`, etc.

For single-target quick checks, use `quick-eval` with auto-logging (requires `--baseline`):

```
uv run fbm quick-eval playing_time --target <target> --inject <candidate> \
  --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --baseline <baseline_rmse> \
  --experiment "<hypothesis>" --tags "experiment-skill,<category>"
```

**Interpret results:**
- `avg_delta_pct` < 0 means improvement (lower RMSE)
- Playing-time has a single target per player type, so interpretation is straightforward
- Focus on consistency across holdout seasons — a feature that helps 2024 but hurts 2023 may be overfitting

### Step 4 — Iterate

After each experiment:
1. Review the result — did it match the hypothesis?
2. If the feature helped, consider:
   - Interactions with other features (`fbm feature interact`)
   - Binned versions (`fbm feature bin`)
   - Non-linear transforms (squared, log) if the relationship isn't linear
3. If the feature didn't help, consider:
   - Why not? Was the signal absorbed by existing features (e.g., `pt_trend` already captures trajectory)?
   - Does the residual analysis suggest a different angle?
4. Update the diagnosis if needed — re-run residuals with new context

### Step 5 — Validate winners

After completing the experiment budget, if any candidates showed consistent improvement (negative `avg_delta_pct`):

```
# Compare full feature set A (default) vs B (default + winners)
uv run fbm compare-features playing_time \
  --set-a default \
  --set-b "<default columns>,<winner1>,<winner2>" \
  --player-type <type> \
  --season 2019 --season 2021 --season 2022 --season 2023 --season 2024
```

If the combined set improves, checkpoint it:

```
uv run fbm experiment checkpoint save "<descriptive-name>" \
  --from-experiment <best_experiment_id> \
  --model playing_time \
  --player-type <type> \
  --notes "<summary of what improved and by how much>"
```

## Summary report

After all iterations, present a summary to the user:

1. **Experiments run:** N total (X positive, Y negative)
2. **Best candidates:** List features that showed improvement, with delta_pct
3. **Worst ideas:** List features that were expected to help but didn't (and why)
4. **Patterns observed:** What types of features tend to help this model?
5. **Recommended next steps:** Suggestions for further exploration based on patterns found
6. **Checkpoint:** Name of saved checkpoint if a winning feature set was identified

## Rules

- **Never skip logging.** Every experiment gets logged, even failures. Negative results prevent duplicate work.
- **No correlation screening gate.** OLS training is fast enough to skip the `feature candidate --correlate` step. Go directly to marginal-value testing.
- **Respect the budget.** Stop at N iterations even if you have more ideas. Present remaining ideas as recommendations.
- **Don't re-test known features.** Always check the experiment journal first.
- **Be specific in hypotheses.** "Maybe age helps" is not a hypothesis. "Age-squared captures non-linear decline in playing time for players over 35 because teams prefer younger alternatives" is.
- **Report honestly.** If nothing improved, say so. Don't overstate marginal gains.
- **Use 2020 exclusion.** Always exclude the 2020 shortened season from training and evaluation seasons.
- **Stay within FanGraphs data.** Don't propose statcast SQL aggregations — playing-time features come from fangraphs tables and derived transforms, not the statcast pitch table.
