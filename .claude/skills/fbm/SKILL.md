---
name: fbm
description: Run fantasy baseball projection commands — predict, evaluate, compare systems, look up player projections/valuations, manage cached datasets, evaluate keeper league trades and optimization, run draft boards and mock drafts, and use Yahoo Fantasy integration for rosters, draft tracking, and keeper cost derivation. Use this when the user asks to run projections, compare systems, evaluate accuracy, look up a player, check valuations, manage/rebuild cached datasets, evaluate keeper league decisions/trades/optimization, view draft boards, run mock drafts, sync Yahoo league data, or manage Yahoo-derived keeper costs.
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

### Look up a player's valuations
```
uv run fbm valuations lookup "<player name>" --season <year> [--system <system>]
```
Example: `uv run fbm valuations lookup "Soto" --season 2025`

### Show valuation rankings
```
uv run fbm valuations rankings --season <year> [--system <system>] [--player-type <type>] [--position <pos>] [--top <N>]
```
Example: `uv run fbm valuations rankings --season 2025 --top 20`

### Evaluate valuation accuracy
Compare predicted auction dollar values against end-of-season actuals. Reports value MAE and Spearman rank correlation.
```
uv run fbm valuations evaluate --season <year> --league <league-name> [--system <system>] [--version <version>] [--top <N>]
```
Example: `uv run fbm valuations evaluate --season 2025 --league default`

The `--league` flag refers to a named league defined in `fbm.toml` under `[leagues.<name>]`. The `--system` defaults to `zar` and `--version` defaults to `1.0`.

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

### Keeper league management

Import keeper costs from CSV:
```
uv run fbm keeper import <csv-path> --season <year> --league <name> [--source <type>] [--format auction|draft-pick]
```
CSV columns: `Player`, `Cost`, `Years` (Years optional, defaults to 1). Use `--format draft-pick` with `--system` and `--provider` for round-based costs.

Set a single player's keeper cost:
```
uv run fbm keeper set "<player name>" --cost <n> --season <year> --league <name> [--years <n>] [--source <type>]
```
Or by draft round: `uv run fbm keeper set "<player name>" --round <n> --season <year> --league <name> --system <system> --provider <provider>`

Show keeper decisions ranked by surplus:
```
uv run fbm keeper decisions --season <year> --league <name> --system <system> [--threshold <n>] [--decay <n>]
```

Show post-keeper adjusted rankings (recalculated replacement levels):
```
uv run fbm keeper adjusted-rankings --season <year> --league <name> --system <system> [--top <N>] [--threshold <n>] [--decay <n>]
```

Evaluate a trade using surplus value:
```
uv run fbm keeper trade-eval --gives "<player>" [--gives "<player>"] --receives "<player>" [--receives "<player>"] --season <year> --league <name> --system <system> [--decay <n>]
```
Example: `uv run fbm keeper trade-eval --gives "Mike Trout" --receives "Shohei Ohtani" --season 2026 --league dynasty --system zar`

Repeat `--gives` / `--receives` for multi-player trades.

Find the optimal keeper set maximizing total surplus:
```
uv run fbm keeper optimize --season <year> --league <name> --system <system> --max-keepers <n> [--max-per-position <pos>=<n>]... [--max-cost <n>] [--required "<player>"]... [--league-keepers <csv>] [--round-escalation <n>] [--max-per-round <n>] [--protected-rounds <n>]... [--threshold <n>] [--decay <n>]
```
Example: `uv run fbm keeper optimize --season 2026 --league dynasty --system zar --max-keepers 8 --max-per-position c=1`

Compare named keeper scenarios side-by-side:
```
uv run fbm keeper scenario --season <year> --league <name> --system <system> --max-keepers <n> --scenario "Name:Player1,Player2" [--scenario "Name2:Player3,Player4"] [--threshold <n>] [--decay <n>]
```
Example: `uv run fbm keeper scenario --season 2026 --league dynasty --system zar --max-keepers 8 --scenario "Plan A:Soto,Ohtani" --scenario "Plan B:Soto,Acuna"`

Show how acquiring/releasing players changes the optimal keeper set:
```
uv run fbm keeper trade-impact --season <year> --league <name> --system <system> --max-keepers <n> [--acquire "<player>"]... [--release "<player>"]... [--acquire-cost <n>]... [--threshold <n>] [--decay <n>]
```
Example: `uv run fbm keeper trade-impact --season 2026 --league dynasty --system zar --max-keepers 8 --acquire "Soto" --acquire-cost 45 --release "Acuna"`

### Draft board and analysis

Display the draft board:
```
uv run fbm draft board --season <year> [--system <system>] [--league <name>] [--player-type <type>] [--position <pos>] [--top <N>] [--provider <adp-provider>]
```
Example: `uv run fbm draft board --season 2026 --top 50`

Export draft board to CSV or HTML:
```
uv run fbm draft export <output-path> --season <year> [--format csv|html] [--system <system>] [--league <name>] [--top <N>]
```

Start a live draft server (auto-refreshing HTML board):
```
uv run fbm draft live --season <year> [--system <system>] [--league <name>] [--host <host>] [--port <port>] [--top <N>]
```

Start an interactive draft session (REPL):
```
uv run fbm draft start --season <year> --teams <n> --slot <n> [--format snake|auction] [--system <system>] [--league <name>] [--budget <n>] [--resume <draft-file>]
```
Example: `uv run fbm draft start --season 2026 --teams 12 --slot 5 --format snake`

Generate a post-draft analysis report:
```
uv run fbm draft report <draft-file> [--season <year>] [--system <system>] [--league <name>]
```

Display position-grouped tier assignments:
```
uv run fbm draft tiers --season <year> [--system <system>] [--method gap|jenks] [--max-tiers <n>] [--position <pos>]
```

Display cross-position tier summary matrix:
```
uv run fbm draft tier-summary --season <year> [--system <system>] [--method gap|jenks] [--max-tiers <n>]
```

Identify category weaknesses and recommend players:
```
uv run fbm draft needs --roster "<player1>,<player2>,..." --season <year> [--system <system>] [--league <name>]
```

Re-rank available players by marginal value given your roster:
```
uv run fbm draft upgrades --season <year> --roster "<player1>,<player2>,..." [--system <system>] [--league <name>] [--top <N>] [--opportunity-cost] [--picks-until-next <n>]
```
Or use `--roster-file <path>` instead of `--roster`.

Show per-position upgrade comparison for your roster:
```
uv run fbm draft position-check --season <year> --roster "<player1>,<player2>,..." [--system <system>] [--league <name>]
```

Display positional scarcity analysis:
```
uv run fbm draft scarcity --season <year> [--system <system>] [--league <name>] [--position <pos>] [--detail]
```

Display scarcity-adjusted player rankings:
```
uv run fbm draft scarcity-rankings --season <year> [--system <system>] [--league <name>] [--top <N>]
```

Display pick value curve:
```
uv run fbm draft pick-values --season <year> [--system <system>] [--provider <adp-provider>] [--league <name>]
```

Evaluate a draft pick trade:
```
uv run fbm draft trade-picks --gives <pick1>,<pick2> --receives <pick3>,<pick4> --season <year> [--system <system>] [--league <name>] [--cascade] [--team <n>] [--threshold <n>]
```
Example: `uv run fbm draft trade-picks --gives 15,30 --receives 8 --season 2026`

### Mock draft simulations

Run a single mock draft:
```
uv run fbm draft mock single --season <year> [--system <system>] [--league <name>] [--teams <n>] [--position <n>] [--strategy <strategy>] [--seed <n>]
```
Example: `uv run fbm draft mock single --season 2026 --position 3 --strategy best-value`

Run batch mock draft simulations:
```
uv run fbm draft mock batch --season <year> [--simulations <n>] [--system <system>] [--league <name>] [--teams <n>] [--position <n>] [--strategy <strategy>] [--seed <n>]
```
Example: `uv run fbm draft mock batch --season 2026 --simulations 500 --strategy best-value`

Compare multiple draft strategies:
```
uv run fbm draft mock compare --season <year> [--strategies <s1>,<s2>,...] [--simulations <n>] [--system <system>] [--league <name>] [--position <n>]
```
Example: `uv run fbm draft mock compare --season 2026 --strategies adp,best-value,positional-need --simulations 200`

Available strategies: `adp`, `best-value`, `positional-need`, `random`.

### Yahoo Fantasy integration

Authenticate with Yahoo Fantasy API (opens browser for OAuth):
```
uv run fbm yahoo auth [--config-dir <dir>]
```

Sync league metadata from Yahoo:
```
uv run fbm yahoo sync --league <name> [--season <year>] [--config-dir <dir>]
```

Display all teams' rosters from Yahoo:
```
uv run fbm yahoo rosters --league <name> [--season <year>]
```

Show your roster with projections and valuations:
```
uv run fbm yahoo my-roster --league <name> [--season <year>]
```

Manually map a Yahoo player key to an internal player:
```
uv run fbm yahoo map-player <yahoo-key> "<player-name>" [--player-type batter|pitcher]
```

Fetch and display draft results from Yahoo:
```
uv run fbm yahoo draft-results --league <name> [--season <year>]
```

Start a live Yahoo draft session with auto-pick ingestion:
```
uv run fbm yahoo draft-live --league <name> [--season <year>] [--league-config <name>] [--system <system>] [--poll-interval <seconds>]
```

Fetch and display recent league transactions:
```
uv run fbm yahoo transactions --league <name> [--season <year>] [--days <n>]
```

Incrementally sync all league data (rosters + transactions):
```
uv run fbm yahoo refresh --league <name> [--season <year>]
```

Derive keeper costs from Yahoo draft history:
```
uv run fbm yahoo keeper-costs --league <name> [--season <year>] [--cost-floor <n>]
```

Show keeper decisions using Yahoo-derived costs:
```
uv run fbm yahoo keeper-decisions --league <name> [--season <year>] [--system <system>] [--threshold <n>] [--decay <n>] [--cost-floor <n>]
```

Show a player's keeper cost history across seasons:
```
uv run fbm yahoo keeper-history "<player-name>" --league <name>
```

Manually set a keeper cost for a Yahoo league player:
```
uv run fbm yahoo keeper-cost-set "<player-name>" --cost <n> --league <name> [--season <year>] [--years <n>] [--source <type>]
```

### Model run inspection

Inspect the latest (or a specific) model run with structured output:
```
uv run fbm runs inspect <system> [--version <version>] [--section config|metrics|tags] [--operation <op>]
```
Example: `uv run fbm runs inspect statcast-gbm` (shows latest train run)
Example: `uv run fbm runs inspect statcast-gbm --version 2026.1 --section metrics`

Compare two model runs side by side:
```
uv run fbm runs diff <run_a> <run_b> [--operation <op>]
```
Each argument is `system/version` or just `system` (uses latest).
Example: `uv run fbm runs diff statcast-gbm/2026.1 statcast-gbm/2026.2`
Example: `uv run fbm runs diff statcast-gbm marcel` (compares latest of each)

## Argument mapping

When the user says something like:
- "compare marcel, steamer, and zips for 2025" → `uv run fbm compare marcel/latest steamer/2025 zips/2025 --season 2025`
- "compare systems with top 300" → add `--top 300`
- "run marcel projections for 2025" → `uv run fbm predict marcel --season 2022 --season 2023 --season 2024` (marcel needs 3 prior seasons)
- "how did marcel do in 2025" → `uv run fbm evaluate marcel --season 2025`
- "look up Ohtani" → `uv run fbm projections lookup "Ohtani" --season 2026`
- "list datasets" → `uv run fbm datasets list`
- "rebuild statcast-gbm features" → `uv run fbm datasets rebuild statcast-gbm --yes`
- "drop all cached datasets" → `uv run fbm datasets drop --all --yes`
- "what's Soto's auction value" → `uv run fbm valuations lookup "Soto" --season 2026`
- "show top 20 valuations" → `uv run fbm valuations rankings --season 2026 --top 20`
- "top pitcher valuations" → `uv run fbm valuations rankings --season 2026 --player-type pitcher --top 20`
- "how accurate were the valuations" → `uv run fbm valuations evaluate --season 2025 --league default`
- "who should I keep" → `uv run fbm keeper decisions --season 2026 --league dynasty --system zar`
- "should I trade Trout for Ohtani" → `uv run fbm keeper trade-eval --gives "Trout" --receives "Ohtani" --season 2026 --league dynasty --system zar`
- "set Trout's keeper cost to $25" → `uv run fbm keeper set "Trout" --cost 25 --season 2026 --league dynasty`
- "adjusted rankings after keepers" → `uv run fbm keeper adjusted-rankings --season 2026 --league dynasty --system zar --top 50`
- "optimize my keepers" → `uv run fbm keeper optimize --season 2026 --league dynasty --system zar --max-keepers 8`
- "compare keeper scenarios" → `uv run fbm keeper scenario --season 2026 --league dynasty --system zar --max-keepers 8 --scenario "Plan A:Soto,Ohtani" --scenario "Plan B:Soto,Acuna"`
- "what if I trade for Soto at $40" → `uv run fbm keeper trade-impact --season 2026 --league dynasty --system zar --max-keepers 8 --acquire "Soto" --acquire-cost 40`
- "show me the draft board" → `uv run fbm draft board --season 2026 --top 50`
- "export draft board to CSV" → `uv run fbm draft export board.csv --season 2026`
- "show draft tiers" → `uv run fbm draft tiers --season 2026`
- "what positions are scarce" → `uv run fbm draft scarcity --season 2026`
- "scarcity-adjusted rankings" → `uv run fbm draft scarcity-rankings --season 2026 --top 50`
- "start a mock draft" → `uv run fbm draft mock single --season 2026`
- "run 500 mock drafts comparing strategies" → `uv run fbm draft mock compare --season 2026 --simulations 500`
- "start a live draft" → `uv run fbm draft start --season 2026 --teams 12 --slot 5`
- "evaluate trading pick 15 for pick 8" → `uv run fbm draft trade-picks --gives 15 --receives 8 --season 2026`
- "I need help at catcher, my roster is..." → `uv run fbm draft needs --roster "Player1,Player2,..." --season 2026`
- "rank upgrades for my roster" → `uv run fbm draft upgrades --season 2026 --roster "Player1,Player2,..." --top 10`
- "sync my Yahoo league" → `uv run fbm yahoo sync --league keeper`
- "show my Yahoo roster" → `uv run fbm yahoo my-roster --league keeper`
- "show Yahoo draft results" → `uv run fbm yahoo draft-results --league keeper`
- "start live Yahoo draft" → `uv run fbm yahoo draft-live --league keeper`
- "what are my keeper costs from Yahoo" → `uv run fbm yahoo keeper-costs --league keeper`
- "Yahoo keeper decisions" → `uv run fbm yahoo keeper-decisions --league keeper --system zar`
- "recent transactions" → `uv run fbm yahoo transactions --league keeper --days 7`
- "refresh Yahoo data" → `uv run fbm yahoo refresh --league keeper`
- "inspect statcast-gbm" → `uv run fbm runs inspect statcast-gbm`
- "show latest statcast-gbm run" → `uv run fbm runs inspect statcast-gbm`
- "inspect statcast-gbm metrics" → `uv run fbm runs inspect statcast-gbm --section metrics`
- "diff statcast-gbm runs" → `uv run fbm runs diff statcast-gbm/2026.1 statcast-gbm/2026.2`
- "compare model runs" → `uv run fbm runs diff <system/version_a> <system/version_b>`

For `--league` in yahoo commands, use the league name from `[yahoo.leagues]` in `fbm.toml` (e.g., `keeper`, `redraft`).
For `--league` in non-yahoo commands, use the league name from `[leagues]` in `fbm.toml` (e.g., `default`, `dynasty`).

For third-party systems (steamer, zips, atc), the version is typically the season year (e.g. `steamer/2025`).
For first-party models (marcel), the version is typically `latest`.

## Output

After running the command, summarize the results for the user in a clear, readable format. Highlight key takeaways and notable comparisons.

If $ARGUMENTS is provided, map it to the appropriate subcommand and run it. Otherwise, ask the user what they'd like to do.

## Interpreting evaluation results

### Before/after protocol

After any model training or tuning change, always run **both** commands before declaring the change an improvement:

```
uv run fbm compare old/ver new/ver --season YEAR --top 300 --check
uv run fbm compare old/ver new/ver --season YEAR
```

Run on at least two holdout seasons. Both the top-300 (fantasy-relevant subset) and full-population comparisons must pass. If either `--check` invocation exits non-zero on any season, the change is a regression.

### Reading the summary

The summary footer line and delta columns (`Δ`, `%Δ`) are the primary signal. Do not override them with per-stat cherry-picking — a change that helps one stat but hurts five is a regression regardless of how important the one stat seems. The `--check` flag's pass/fail verdict is authoritative.

### What "better" means

- **RMSE**: lower is better.
- **R²**: higher is better.
- **Rank correlation (ρ)**: higher is better.

A change must win a **majority** of stats to be considered an improvement. Winning one stat while losing several others is a regression, not an improvement.

### Tail accuracy

Use `--tail` to check top-25 and top-50 RMSE. A model that improves full-cohort RMSE but degrades top-N accuracy may not be useful for fantasy — the best players are the most important to predict correctly.

### Common pitfalls

- **Single-stat or single-season success**: always check multiple stats across multiple seasons before declaring improvement.
- **Confusing diagnosis with cure**: identifying a problem in the diagnostic output does not mean the next change fixed it — re-run the comparison to verify.
- **Ignoring full-population regressions**: always run without `--top` as well. Regressions outside the top-300 can indicate overfitting.
- **Ignoring rank correlation**: a model with lower RMSE but worse player ordering (lower ρ) is worse for fantasy, where correct rankings matter more than absolute accuracy.
