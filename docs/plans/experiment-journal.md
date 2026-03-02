# Experiment Journal

Give an autonomous agent persistent memory of what it has tried, what worked, and what didn't. Without this, the agent will re-try failed ideas, lose track of promising directions, and lack the context to build on prior findings across sessions. The journal is the difference between systematic exploration and random wandering.

## Status

| Phase | Status |
|-------|--------|
| 1 — Experiment logger | done (2026-03-02) |
| 2 — Experiment query tool | done (2026-03-02) |
| 3 — Checkpoint / restore | not started |

## Phase 1: Experiment logger

Record each feature exploration trial with its hypothesis, configuration, results, and conclusion in a structured, queryable store.

### Context

The agent will run dozens or hundreds of trials — testing new features, feature combinations, hyperparameter tweaks, and training strategies. Each trial has a hypothesis ("adding barrel rate will improve SLG"), a configuration (feature set, seasons, params), and a result (per-target RMSE deltas). Without structured logging, this knowledge is lost between sessions.

### Steps

1. Create `src/fantasy_baseball_manager/domain/experiment.py` with an `Experiment` frozen dataclass: `id` (auto-generated UUID), `timestamp`, `hypothesis` (free text), `model` (system name), `player_type`, `feature_diff` (columns added/removed vs baseline), `seasons` (train + holdout), `params` (hyperparameters used), `target_results` (dict of target → `TargetResult` with rmse, baseline_rmse, delta, delta_pct), `conclusion` (free text), `tags` (list of strings for categorization), `parent_id` (optional, for building on a prior experiment).
2. Create `src/fantasy_baseball_manager/repos/experiment_repo.py` with a SQLite-backed repo implementing `save(experiment)`, `get(id)`, `list(filters)`, `delete(id)`.
3. Add an `experiment` table to the main database via a migration: all fields from the dataclass, with indexes on `model`, `tags`, and `timestamp`.
4. Add a `fbm experiment log --hypothesis "..." --model statcast-gbm-preseason --feature-diff "+barrel_ev" --target-results '{"slg": {"rmse": 0.082, "baseline": 0.085, "delta": -0.003}}' --conclusion "..." --tags feature,batter,slg` CLI command.
5. Support auto-logging from the quick-eval and marginal-value tools: if an `--experiment` flag is passed, the result is automatically saved to the journal with the command's parameters as the configuration.
6. Write tests verifying save/load round-trip, filtering, and auto-logging integration.

### Acceptance criteria

- Experiments are persisted in SQLite and survive across sessions.
- All fields are queryable (by model, tags, date range, etc.).
- Auto-logging from quick-eval tools works without manual entry.
- Parent-child relationships link follow-up experiments to their predecessors.

## Phase 2: Experiment query tool

Search and summarize past experiments to answer questions like "what features have I tried for pitcher ERA?" or "what was the best feature set I found?"

### Context

Logging experiments is only useful if the agent can query them efficiently. The agent needs to ask natural questions about its history: what worked for a specific target, what features have been tested, what's the best result so far, and whether a specific idea has already been tried.

### Steps

1. Add query functions to the experiment repo: `find_by_target(target)`, `find_by_tag(tag)`, `find_by_feature(column_name)`, `find_best(target, metric="delta_pct")`, `find_by_model(model)`, `find_recent(n)`.
2. Add a summary function `summarize_exploration(model, player_type)` that returns: total experiments run, features tested (with best result per feature), targets explored (with best RMSE achieved), and the overall best configuration found.
3. Add a `fbm experiment search --target slg --tag feature --model statcast-gbm-preseason` CLI command that prints matching experiments sorted by result quality.
4. Add a `fbm experiment summary --model statcast-gbm-preseason --player-type batter` CLI command that prints the exploration summary.
5. Add a `fbm experiment show <id>` CLI command for viewing a single experiment's full details.
6. Write tests verifying all query patterns return correct results.

### Acceptance criteria

- All query patterns return correct, filtered results.
- Summary correctly aggregates across experiments to show the best finding per target and per feature.
- Duplicate detection works: searching by feature name finds all experiments that tested that feature.
- Results are sorted by quality (best delta first).

## Phase 3: Checkpoint / restore

Save a promising feature set configuration as a named checkpoint so the agent can branch from it later.

### Context

As the agent explores, it will find a feature set that's better than baseline. It should be able to save this as a checkpoint ("best_batter_v3"), continue exploring from it, and return to it if later experiments make things worse. This is like git branches for feature sets.

### Steps

1. Create `src/fantasy_baseball_manager/domain/checkpoint.py` with a `FeatureCheckpoint` frozen dataclass: `name`, `model`, `player_type`, `feature_columns` (list of column names), `params` (hyperparameters), `target_results` (per-target metrics at checkpoint time), `experiment_id` (link to the experiment that produced this), `created_at`, `notes`.
2. Add checkpoint functions to the experiment repo: `save_checkpoint(checkpoint)`, `get_checkpoint(name)`, `list_checkpoints(model)`, `delete_checkpoint(name)`.
3. Store checkpoints in a `feature_checkpoint` table with a UNIQUE constraint on `(name, model)`.
4. Add a `fbm experiment checkpoint save <name> --model statcast-gbm-preseason --from-experiment <id>` CLI command that saves the feature set from a specific experiment as a named checkpoint.
5. Add a `fbm experiment checkpoint list --model statcast-gbm-preseason` CLI command.
6. Add a `fbm experiment checkpoint restore <name>` CLI command that prints the feature set and params, ready to use in other tools (e.g., `--set-a checkpoint:best_batter_v3`).
7. Integrate with the fast-feedback-loop tools: `--set-a checkpoint:<name>` loads a checkpoint's feature set as the A set in comparisons.
8. Write tests verifying save/load round-trip and integration with comparison tools.

### Acceptance criteria

- Checkpoints persist across sessions.
- Restoring a checkpoint returns the exact feature set and params.
- Checkpoints are linked to their source experiment.
- Integration with comparison tools works via the `checkpoint:<name>` syntax.
- Duplicate checkpoint names for the same model are rejected (upsert or error).

## Ordering

Phases are sequential: 1 → 2 → 3. Phase 1 provides the core logging infrastructure. Phase 2 adds querying on top of logged data. Phase 3 adds named checkpoints for branching. Phase 1 is useful standalone — even without querying, having a persistent log is better than nothing.

## Dependencies

- **fast-feedback-loop**: Auto-logging in phase 1 integrates with quick-eval and marginal-value tools. Checkpoint restore in phase 3 integrates with compare-features. Without these, experiments must be logged and restored manually.
