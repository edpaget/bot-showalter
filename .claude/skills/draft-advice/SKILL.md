---
user-invocable: true
description: Get fantasy baseball draft advice, player projections, and team comparisons
---

# Skill: Draft Advice

Provide fantasy baseball draft advice, player analysis, and team comparison using the `fantasy-baseball-manager` CLI.

## Gathering Context

Before running commands, determine:

1. **Year**: Default to the current year. Ask if unclear.
2. **Engine**: Default to `marcel_plus` (most accurate). Options: `marcel`, `marcel_park`, `marcel_statreg`, `marcel_plus`.
3. **Yahoo integration**: Commands like `teams compare`, `teams roster`, and `players draft-rank --yahoo` require Yahoo Fantasy API access. Use `--yahoo` when the user wants live draft data or team rosters.
4. **Category weights**: If the user's league values certain categories more, use `--weight` flags (e.g., `--weight HR=2.0 --weight SB=1.5`).

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
