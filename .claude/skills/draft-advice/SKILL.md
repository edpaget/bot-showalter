---
user-invocable: true
description: Get fantasy baseball draft advice, player projections, and team comparisons
---

# Skill: Draft Advice

Provide fantasy baseball draft advice, player analysis, and team comparison using the `fantasy-baseball-manager` CLI.

## League Context

Scoring categories are configured in `config.yaml` under `league.batting_categories` and `league.pitching_categories`. All CLI commands automatically use these categories — no need to specify them manually. The `--categories` flag on `players valuate` is only needed to override the config for a one-off query. The `--weight` flag adjusts category multipliers without changing which categories are used.

## Gathering Context

Before running commands, determine:

1. **Year**: Default to the current year. Ask if unclear.
2. **Engine**: Default to `marcel_plus` (most accurate). Options: `marcel`, `marcel_park`, `marcel_statreg`, `marcel_plus`.
3. **Yahoo integration**: Commands like `teams compare`, `teams roster`, and `players draft-rank --yahoo` require Yahoo Fantasy API access. Use `--yahoo` when the user wants live draft data or team rosters.
4. **Category weights**: If the user's league values certain categories more, use `--weight` flags (e.g., `--weight HR=2.0 --weight SB=1.5`). These are multipliers on top of the categories already configured in `config.yaml`.

## Command Routing

Map user intent to the right CLI subcommand:

| User Intent | Command |
|---|---|
| "Who should I draft?" / best available | `uv run fantasy-baseball-manager players draft-rank --engine marcel_plus --top 30` |
| Best available batters | `uv run fantasy-baseball-manager players draft-rank --engine marcel_plus --top 30 --batting` |
| Best available pitchers | `uv run fantasy-baseball-manager players draft-rank --engine marcel_plus --top 30 --pitching` |
| Draft rankings with Yahoo positions/draft state | Add `--yahoo` to any `players draft-rank` command |
| Raw player projections | `uv run fantasy-baseball-manager players project --engine marcel_plus --top 30` |
| Z-score valuations | `uv run fantasy-baseball-manager players valuate --engine marcel_plus --top 30` |
| Compare all teams in league | `uv run fantasy-baseball-manager teams compare --engine marcel_plus` |
| My team's roster projections | `uv run fantasy-baseball-manager teams roster --engine marcel_plus` |
| Projection accuracy / backtesting | `uv run fantasy-baseball-manager evaluate 2025 --engine marcel_plus` |
| Simulate a full draft | `uv run fantasy-baseball-manager players draft-simulate --teams 12 --seed 42` |
| Simulate with a specific strategy | `uv run fantasy-baseball-manager players draft-simulate --user-strategy power_hitting --seed 42` |
| Simulate with keepers | `uv run fantasy-baseball-manager players draft-simulate --keepers keepers.yaml --seed 42` |
| Rank keeper candidates | `uv run fantasy-baseball-manager keeper rank --yahoo --engine marcel_plus` |
| Optimize keeper selection | `uv run fantasy-baseball-manager keeper optimize --yahoo --engine marcel_plus` |
| League-wide keeper analysis | `uv run fantasy-baseball-manager keeper league --engine marcel_plus` |

Adjust `--top` based on how many results the user wants. Use `--sort-by` for custom sorting (e.g., `--sort-by hr`, `--sort-by era`).

## Output Formats

### `players draft-rank`

Fixed-width table with columns:

- **Rk**: Rank (the draft position recommendation)
- **Name**: Player name (25 chars)
- **Pos**: Eligible positions, slash-separated (e.g., `SS/2B`)
- **Mult**: Positional scarcity multiplier (higher = scarcer position)
- **Raw**: Raw z-score value before weighting
- **Wtd**: Weighted value (after category weight adjustments)
- **Adj**: Final adjusted value (Wtd * Mult) -- this is the primary ranking metric

### `players project`

Batting columns: Name, Age, PA, HR, R, RBI, AVG, OBP, SB
Pitching columns: Name, Age, IP, ERA, WHIP, SO, W, NSVH, HR

### `players valuate`

Columns: Name, then one column per scoring category, then Total z-score.

### `teams compare`

Columns: Team (25 chars), HR, SB, AVG, OBP, IP, SO, ERA, WHIP, ? (unmatched player count)

The `?` column shows how many rostered players couldn't be matched to projections. High numbers mean less reliable team totals.

### `teams roster`

Per-team breakdown showing:
- Batter table: Name, PA, HR, AVG, OBP, SB
- Pitcher table: Name, IP, ERA, WHIP, SO
- Warning line if players couldn't be matched

## Strategy Guidance

When interpreting results and giving advice:

### Value-Based Drafting
- Recommend the player with the highest **Adj** value in `draft-rank` output.
- The Adj value accounts for both statistical production and positional scarcity.

### Positional Scarcity
- Scarce positions (C, SS, 2B) have higher Mult values -- drafting them early locks in value that disappears fast.
- Deep positions (OF, SP, 1B/DH) have lower Mult values -- you can wait on these.
- When two players have similar Adj values, prefer the scarcer position.

### Category Balance
- Use `teams compare` to identify which categories a team is weak in.
- If a team is weak in SB, recommend targeting high-SB players. If weak in ERA/WHIP, recommend targeting pitching.
- Use `--weight` flags to boost underrepresented categories in draft rankings (e.g., `--weight SB=1.5`).

### Comparing Players
- When asked to compare two specific players, run `players project` to see their raw stat lines side-by-side.
- Use `players valuate` to see how their stats translate to category value.
- Use `players draft-rank` to see how positional scarcity affects their overall value.

## Keeper Analysis

When the user wants to evaluate which players to keep in a keeper league, use the `keeper` subcommands. These compute **surplus value** — the difference between a player's projected value and what you'd draft at the corresponding slot.

### Command Routing

| User Intent | Command |
|---|---|
| Rank my keeper candidates | `uv run fantasy-baseball-manager keeper rank --yahoo --engine marcel_plus` |
| Find the optimal keeper combination | `uv run fantasy-baseball-manager keeper optimize --yahoo --engine marcel_plus` |
| Compute optimal keepers for every team | `uv run fantasy-baseball-manager keeper league --engine marcel_plus` |
| Rank specific players as keeper candidates | `uv run fantasy-baseball-manager keeper rank --candidates "id1,id2,id3" --engine marcel_plus` |

### How Surplus Value Works

1. **Replacement calculation**: A snake draft is simulated with the remaining player pool (excluding keepers). For each keeper slot, the system records what player value you'd draft at your pick position.
2. **Surplus**: `Surplus = Player Value - Replacement Value`. Positive surplus means keeping the player is better than drafting at that slot.
3. **Optimization**: `keeper optimize` tests all valid combinations of candidates and picks the set that maximizes total surplus. This matters because removing top players from the draft pool shifts replacement values.

### `keeper rank` / `keeper optimize` options

- `YEAR` (positional): Projection year (default: current year)
- `--candidates ID1,ID2,...`: Comma-separated FanGraphs player IDs (not needed with `--yahoo`)
- `--keepers FILE`: YAML file with other teams' keepers (not needed with `--yahoo`)
- `--user-pick N`: User's draft position, 1-based (default: 5)
- `--teams N`: Number of teams (default: 12, auto-detected with `--yahoo`)
- `--keeper-slots N`: Number of keeper slots per team (default: 4)
- `--engine NAME`: Projection engine (default: marcel_plus)
- `--yahoo`: Fetch candidates from Yahoo roster and other teams' keepers automatically
- `--no-cache`: Bypass cache and fetch fresh data
- `--league-id ID`: Override league ID from config
- `--season YEAR`: Override season from config

### `keeper league` options

- `YEAR` (positional): Projection year (default: current year)
- `--draft-order KEY1,KEY2,...`: Comma-separated team keys defining pick order
- `--teams N`, `--keeper-slots N`, `--engine`, `--no-cache`, `--league-id`, `--season`: Same as above

### Output Format

Columns: Rk, Name (25 chars), Pos (eligible positions), Value (z-score), Repl (replacement value at assigned slot), Surplus (Value - Repl), Slot (assigned keeper slot number).

For `keeper optimize`, two tables are shown: the recommended optimal keepers with their total surplus, then all candidates ranked.

For `keeper league`, each team's optimal keepers are shown with their pick number and total surplus.

### Interpreting Results

- **High surplus** players are clear keeps — their value far exceeds what you'd draft.
- **Negative surplus** means you'd get a better player by drafting at that slot — don't keep.
- **`keeper rank` vs `keeper optimize`**: `rank` assigns each candidate to the next available slot independently. `optimize` finds the globally best combination, which can differ because the draft pool changes depending on which players are kept.
- When using `--yahoo`, the system auto-detects team count and fetches all rosters, so other teams' keepers are excluded from the draft pool automatically.

## Draft Simulation

When the user wants to simulate a draft or test strategies:

1. **Analyze keepers** (if applicable): Use `keeper rank` or `keeper optimize` to evaluate keeper surplus values.
2. **Identify category strengths/weaknesses**: Look at the team's keeper pool to determine which categories are strong or weak.
3. **Recommend a strategy preset**: Based on the analysis, suggest one of: `balanced`, `power_hitting`, `speed`, `pitching_heavy`, `punt_saves`.
4. **Run the simulation**: Use `players draft-simulate` with appropriate flags.

### `players draft-simulate` options

- `--teams N`: Number of teams (default: 12)
- `--user-pick N`: User's draft position, 1-based (default: 1)
- `--user-strategy NAME`: Strategy for user team (default: balanced)
- `--opponent-strategy NAME`: Default strategy for opponents (default: balanced)
- `--keepers FILE`: YAML file with per-team keepers and optional per-team strategies
- `--rounds N`: Total draft rounds (default: 20)
- `--seed N`: Random seed for reproducibility
- `--log`: Show pick-by-pick log
- `--rosters / --no-rosters`: Show final team rosters (default: on)
- `--standings / --no-standings`: Show projected roto standings (default: on)

### Strategy presets

- **balanced**: Equal category weights, max 2 catchers, no RP before round 10
- **power_hitting**: HR=1.5, RBI=1.3, R=1.2 weights, max 2 catchers
- **speed**: SB=2.0, R=1.3 weights, max 1 catcher
- **pitching_heavy**: K=1.4, ERA=1.3, WHIP=1.3, W=1.2, allows 60% pitchers
- **punt_saves**: NSVH=0.0, no RP before round 20

### Keepers YAML format

```yaml
teams:
  1:
    name: "My Team"
    keepers: ["player_id_1", "player_id_2"]
  2:
    name: "Opponent 1"
    strategy: "power_hitting"
    keepers: ["player_id_3"]
```

## Live Draft Mode

When the user indicates they are in an active draft:

- Always use `--yahoo` to get current draft state (who's been picked).
- Keep responses **concise** -- just the recommendation and key reasoning.
- After each pick, re-run `players draft-rank --yahoo` to get updated rankings.
- Do NOT use `--no-cache` -- the caching layer handles draft state appropriately.
- Focus on actionable advice: "Draft [Player] (Adj: X.X) -- best value at a scarce position."

## Engine Descriptions

If the user asks which engine to use:

- **marcel**: Basic Marcel projection. Fast, no external data needed beyond stats.
- **marcel_park**: Adds park factor adjustments (e.g., Coors Field boost).
- **marcel_statreg**: Uses stat-specific regression rates instead of uniform regression.
- **marcel_plus**: Combines park factors + stat-specific regression. Most accurate, recommended default.
