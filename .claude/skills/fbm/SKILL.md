---
name: fbm
description: Run fantasy baseball projection commands — predict, evaluate, compare systems, look up player projections, and manage cached datasets. Use this when the user asks to run projections, compare systems, evaluate accuracy, look up a player, or manage/rebuild cached datasets.
allowed-tools: Bash(uv run fbm *)
argument-hint: <command> [args...]
---

# Fantasy Baseball Manager CLI

Run `fbm` commands for the user via `uv run fbm`. Parse the arguments to determine which subcommand to run.

## Available commands

### Compare projection systems
Compare multiple systems against actuals for a season.
```
uv run fbm compare <system/version>... --season <year> [--top <N>] [--stat <stat>...]
```
Example: `uv run fbm compare marcel/latest steamer/2025 zips/2025 --season 2025 --top 300`

### Evaluate a model
Evaluate a single first-party model's accuracy.
```
uv run fbm evaluate <model> --season <year> [--top <N>]
```
Example: `uv run fbm evaluate marcel --season 2025 --top 300`

### Generate predictions
Run a model to produce projections.
```
uv run fbm predict <model> --season <year>... [--version <version>] [--tag key=value]
```
Example: `uv run fbm predict marcel --season 2022 --season 2023 --season 2024`

### Look up a player's projections
```
uv run fbm projections lookup "<player name>" --season <year> [--system <system>]
```
Example: `uv run fbm projections lookup "Soto" --season 2025`

### List available projection systems
```
uv run fbm projections systems --season <year>
```

### Manage cached datasets

List all cached feature-set datasets:
```
uv run fbm datasets list [--name <name>]
```

Drop a specific feature set's cached datasets:
```
uv run fbm datasets drop --name <name> [--yes]
```

Drop all cached datasets:
```
uv run fbm datasets drop --all [--yes]
```

Rebuild a model's datasets (drop + re-prepare):
```
uv run fbm datasets rebuild <model> [--season <year>...] [--yes]
```

## Argument mapping

When the user says something like:
- "compare marcel, steamer, and zips for 2025" → `uv run fbm compare marcel/latest steamer/2025 zips/2025 --season 2025`
- "compare systems with top 300" → add `--top 300`
- "run marcel projections for 2025" → `uv run fbm predict marcel --season 2022 --season 2023 --season 2024` (marcel needs 3 prior seasons)
- "how did marcel do in 2025" → `uv run fbm evaluate marcel --season 2025`
- "look up Ohtani" → `uv run fbm projections lookup "Ohtani" --season 2025`
- "list datasets" → `uv run fbm datasets list`
- "rebuild statcast-gbm features" → `uv run fbm datasets rebuild statcast-gbm --yes`
- "drop all cached datasets" → `uv run fbm datasets drop --all --yes`

For third-party systems (steamer, zips, atc), the version is typically the season year (e.g. `steamer/2025`).
For first-party models (marcel), the version is typically `latest`.

## Output

After running the command, summarize the results for the user in a clear, readable format. Highlight key takeaways and notable comparisons.

If $ARGUMENTS is provided, map it to the appropriate subcommand and run it. Otherwise, ask the user what they'd like to do.
