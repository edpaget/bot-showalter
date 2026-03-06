---
name: experiment-breakout-bust
description: Run an autonomous feature-engineering experiment loop on the breakout-bust classification model. Tests candidate features against P(breakout) and P(bust) probability targets using marginal-value with auto-logging. Use when the user asks to "experiment on breakout-bust", "improve breakout-bust", or "explore features for breakout".
allowed-tools: Bash(uv run fbm *)
argument-hint: <player-type> [target] [--budget N]
---

# Breakout-Bust Experiment Skill

Run a structured experiment loop to discover features that improve the breakout-bust classifier. This skill orchestrates the full pipeline: diagnose weaknesses, generate hypotheses, test candidates, log results, and validate winners.

## Argument parsing

Parse `$ARGUMENTS` for:
- **player-type** (required): `batter` or `pitcher`
- **target** (optional): a specific target to focus on (`p_breakout` or `p_bust`). If omitted, work across both.
- **--budget N** (optional): maximum number of experiment iterations (default: 10)

## Context

**Model:** `breakout-bust` (HistGradientBoostingClassifier, predicts P(breakout), P(bust), P(neutral))

**Targets:** `p_breakout`, `p_bust` (RMSE on probability predictions vs 0/1 indicators)

**Training seasons:** Use 2019-2024 (excluding 2020). Holdout is always the last season specified.

**Batter features (preseason weighted curated):**
`age`, `pa_1`, `hr_1`, `h_1`, `doubles_1`, `triples_1`, `bb_1`, `so_1`, `sb_1`, `avg_1`, `obp_1`, `slg_1`, `k_pct_1`, `bb_pct_1`, `avg_exit_velo`, `max_exit_velo`, `avg_launch_angle`, `hard_hit_pct`, `gb_pct`, `fb_pct`, `ld_pct`, `sweet_spot_pct`, `exit_velo_p90`, `chase_rate`, `zone_contact_pct`, `whiff_rate`, `swinging_strike_pct`, `called_strike_pct`, `xba`, `xwoba`, `xslg`, `pull_pct`, `oppo_pct`, `center_pct`, `adp_rank`, `adp_pick`

**Pitcher features:** preseason averaged curated columns + `adp_rank`, `adp_pick`

**Key difference from statcast-gbm:** Breakout-bust uses the same preseason weighted/averaged feature sets derived from statcast columns, so candidate features come from the same pool. However, targets are classification-based probabilities (not stat predictions), and there is no correlation screening gate — classification training is fast enough to test directly.

## Experiment loop

Execute this loop up to the budget limit. Each iteration should be purposeful — use prior results to guide the next hypothesis.

### Step 0 — Review prior work

Before generating new hypotheses, check what has already been tried:

```
uv run fbm experiment summary --model breakout-bust --player-type <type>
uv run fbm experiment search --model breakout-bust --feature <any-feature-you-plan-to-test>
```

Do NOT re-test features that have already been explored unless you have a materially different hypothesis.

### Step 1 — Diagnose weaknesses

Identify where the model struggles most:

```
# Worst misses on a specific target
uv run fbm residuals worst-misses breakout-bust/latest --season 2024 --player-type <type> --target <target> --top 20

# Cohort bias — find systematic over/under-prediction
uv run fbm residuals cohort breakout-bust/latest --season 2024 --player-type <type> --target <target> --all-dimensions

# Feature distribution gaps between good and bad predictions
uv run fbm residuals gaps breakout-bust/latest --season 2024 --player-type <type> --target <target> --include-raw
```

Analyze the output to identify patterns:
- Are there cohorts where breakout/bust probabilities are systematically miscalibrated?
- Do worst misses share characteristics not captured by existing features?
- Do feature gaps highlight columns that could differentiate breakout from bust candidates?

### Step 2 — Generate hypotheses

Based on the diagnosis, formulate specific, testable hypotheses. Good hypotheses:
- Address a specific weakness found in Step 1
- Have a plausible causal mechanism linking the feature to breakout/bust outcomes

Examples of feature ideas:
- **ADP-based:** ADP volatility (std dev across rankings sources) predicts bust probability because over-drafted players face regression
- **Consensus gap:** Difference between ADP rank and projection-implied rank signals market mispricing
- **Statcast interactions:** `hard_hit_pct * bb_pct_1` captures players with both power and discipline — breakout candidates
- **Age-performance curves:** `age * slg_1` interaction to capture aging breakout/bust asymmetry
- **Trend features:** Year-over-year delta in key stats (if available) to capture trajectory
- **Plate discipline combos:** `chase_rate * swinging_strike_pct` — high values may predict bust (poor contact approach)

### Step 3 — Test with fast feedback and auto-log

Since breakout-bust classification training is fast, skip the correlation screening gate and test directly with marginal-value:

```
# Single candidate — auto-logs hypothesis, feature diff, and per-target results
uv run fbm marginal-value breakout-bust --candidate <column_or_name> \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --experiment "<hypothesis — what you expect and why>" \
  --tags "experiment-skill,<category>"

# Multiple candidates — each gets its own experiment journal entry
uv run fbm marginal-value breakout-bust --candidate cand1 --candidate cand2 \
  --player-type <type> --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --experiment "<hypothesis>" --tags "experiment-skill,<category>"
```

The `--experiment` flag auto-logs: timestamp, hypothesis, model, player type, feature diff (added candidate), train/holdout seasons, per-target RMSE deltas, and a computed conclusion.

Use `--parent-id <N>` to chain related experiments (e.g., variations of the same idea).

Use descriptive tags: `adp`, `plate-discipline`, `batted-ball`, `interaction`, `age-curve`, `trend`, etc.

For single-target quick checks, use `quick-eval` with auto-logging (requires `--baseline`):

```
uv run fbm quick-eval breakout-bust --target <target> --inject <candidate> \
  --season 2019 --season 2021 --season 2022 --season 2023 --season 2024 \
  --baseline <baseline_rmse> \
  --experiment "<hypothesis>" --tags "experiment-skill,<category>"
```

**Interpret results:**
- `avg_delta_pct` < 0 means improvement (lower RMSE on probability predictions)
- Look at per-target deltas — improvement on the target of interest matters most
- A feature that helps `p_breakout` but significantly hurts `p_bust` should be treated cautiously

### Step 4 — Iterate

After each experiment:
1. Review the result — did it match the hypothesis?
2. If the feature helped, consider:
   - Interactions with other features (`fbm feature interact`)
   - Binned versions (`fbm feature bin`)
   - Variations (different aggregation windows, different thresholds)
3. If the feature didn't help, consider:
   - Why not? Was the signal absorbed by existing features?
   - Does the residual analysis suggest a different angle?
4. Update the diagnosis if needed — re-run residuals with new context

### Step 5 — Validate winners

After completing the experiment budget, if any candidates showed consistent improvement (negative `avg_delta_pct` across targets):

```
# Compare full feature set A (default) vs B (default + winners)
uv run fbm compare-features breakout-bust \
  --set-a default \
  --set-b "<default columns>,<winner1>,<winner2>" \
  --player-type <type> \
  --season 2019 --season 2021 --season 2022 --season 2023 --season 2024
```

If the combined set improves across targets, checkpoint it:

```
uv run fbm experiment checkpoint save "<descriptive-name>" \
  --from-experiment <best_experiment_id> \
  --model breakout-bust \
  --player-type <type> \
  --notes "<summary of what improved and by how much>"
```

## Summary report

After all iterations, present a summary to the user:

1. **Experiments run:** N total (X positive, Y negative)
2. **Best candidates:** List features that showed improvement, with delta_pct per target
3. **Worst ideas:** List features that were expected to help but didn't (and why)
4. **Patterns observed:** What types of features tend to help this model?
5. **Recommended next steps:** Suggestions for further exploration based on patterns found
6. **Checkpoint:** Name of saved checkpoint if a winning feature set was identified

## Rules

- **Never skip logging.** Every experiment gets logged, even failures. Negative results prevent duplicate work.
- **No correlation screening gate.** Unlike statcast-gbm, breakout-bust classification training is fast enough to skip the `feature candidate --correlate` step. Go directly to marginal-value testing.
- **Respect the budget.** Stop at N iterations even if you have more ideas. Present remaining ideas as recommendations.
- **Don't re-test known features.** Always check the experiment journal first.
- **Be specific in hypotheses.** "Maybe ADP helps" is not a hypothesis. "ADP volatility (cross-source std dev) should predict bust probability because over-drafted players face regression pressure" is.
- **Report honestly.** If nothing improved, say so. Don't cherry-pick one target's improvement while ignoring regressions on the other.
- **Use 2020 exclusion.** Always exclude the 2020 shortened season from training and evaluation seasons.
