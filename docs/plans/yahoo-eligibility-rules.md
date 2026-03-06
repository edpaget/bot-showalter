# Yahoo Eligibility Rules Roadmap

Align position eligibility thresholds and multi-season logic with Yahoo Fantasy rules.
Currently the system uses a flat `min_games=10` threshold on single-season data with a
binary fallback (use season-1 only when the target season has zero records). This causes
two problems: (1) batters who haven't yet reached 10 games at a position mid-season lose
eligibility they'd have on Yahoo, and (2) prior-season eligibility doesn't carry forward,
so players like Kris Bryant (12 RF games in 2024, none in 2025) lose position assignments
entirely.

Yahoo rules are: batters need 5 games started or 10 games played at a position; pitchers
need 3 starts for SP or 5 relief appearances for RP; and eligibility earned in the prior
season carries into the current season. Because our Lahman-sourced position data doesn't
distinguish games started from games played, we approximate "5 GS or 10 GP" as `>= 5 GP`
(since most non-DH position games involve starting). For pitchers, we already have `g` and
`gs` from `PitchingStats` and just need to tighten the thresholds from `> 0` to the Yahoo
values. The multi-season carryover is the biggest impact: combining current + prior season
data recovers ~110 additional players with legitimate position eligibility.

## Status

| Phase | Status |
|-------|--------|
| 1 — Multi-season eligibility and Yahoo thresholds | done (2026-03-05) |
| 2 — Configurable eligibility rules on LeagueSettings | not started |

## Phase 1: Multi-season eligibility and Yahoo thresholds

Change `PlayerEligibilityService` to combine position data from the current season and
the prior season (matching Yahoo's carryover rule), lower the default batter threshold
from 10 to 5, and tighten pitcher SP/RP thresholds to match Yahoo's 3-start / 5-relief
rules.

### Context

The current eligibility service fetches position data for one season, falling back to
`season - 1` only when the target season has zero records. This means:

- Mid-season, batters who have 5-9 games at a position are excluded (Yahoo would grant
  eligibility at 5 GS).
- Players who earned eligibility last year but haven't played the position this year
  (e.g., Jesse Winker with 65 LF games in 2024 but only 2 in 2025) lose it entirely.
- Pitcher SP/RP classification uses `gs > 0` / `(g - gs) > 0` — no minimum threshold.
  This means a starter who made 1 emergency relief appearance gains RP eligibility, and
  a reliever who made 1 spot start gains SP eligibility.

Impact: only 122 players have positive valuations instead of the expected ~204 (108
batter + 96 pitcher roster spots across 12 teams). The largest cause is ~530 batters
landing in util-only, where the replacement level is set by the 12th-best player across
all ~1,080 batters.

### Steps

1. Change `PlayerEligibilityService.get_batter_positions` to fetch appearances for
   **both** `season` and `season - 1`, combine them (union of eligible positions across
   both years), and pass the merged list to `build_position_map`. The prior-season data
   provides carryover eligibility; current-season data adds newly earned positions. Each
   season's appearances are filtered by `min_games` independently before merging.
2. Change the default `min_games` from 10 to 5 in `get_batter_positions`.
3. Change `PlayerEligibilityService.get_pitcher_positions` to use `gs >= 3` for SP
   eligibility and `(g - gs) >= 5` for RP eligibility (replacing `gs > 0` / `(g-gs) > 0`).
4. Apply the same multi-season logic to pitcher classification: combine stats from
   `season` and `season - 1` (take max `g`, max `gs` across both seasons per player).
5. Update existing tests for the new defaults and multi-season behavior. Add new tests:
   - Batter with position in prior season only (carryover).
   - Batter with position in both seasons (union of positions).
   - Pitcher with 2 starts (below SP threshold) — should not get SP.
   - Pitcher with 4 relief apps (below RP threshold) — should not get RP.
   - Pitcher with stats in prior season only (carryover).
   - Preseason scenario (no data for target season): prior season used alone.
6. Regenerate `zar latest` valuations and verify the positive-player count is in the
   expected range (~170-210).

### Acceptance criteria

- Batter eligibility combines current + prior season data (union of positions).
- Default batter `min_games` threshold is 5.
- Pitcher SP eligibility requires `gs >= 3`; RP requires `(g - gs) >= 5`.
- Pitcher classification also uses multi-season data.
- Players like Kyle Schwarber (8 LF games in 2025, plus prior-year eligibility) get
  correct OF eligibility.
- Players like Kris Bryant (12 RF in 2024, 0 in 2025) retain OF eligibility via
  carryover.
- Starters with 1-2 relief appearances do NOT gain RP eligibility.
- Relievers with 1-2 spot starts do NOT gain SP eligibility.
- All existing tests updated and passing.

## Phase 2: Configurable eligibility rules on LeagueSettings

Make the eligibility thresholds configurable per league, so different platforms' rules
can be expressed in `fbm.toml`.

### Context

Phase 1 hardcodes Yahoo's specific thresholds. Other platforms (ESPN, CBS, etc.) use
different rules. Making these configurable lets the system support multiple league
platforms without code changes.

### Steps

1. Add an `eligibility` field to `LeagueSettings` — a frozen dataclass
   `EligibilityRules` with fields: `batter_min_games` (default 5),
   `sp_min_starts` (default 3), `rp_min_relief` (default 5),
   `carryover_seasons` (default 1 — how many prior seasons to include).
2. Parse the `[leagues.h2h.eligibility]` section in `config_league.py`. All fields
   optional with defaults matching Yahoo rules.
3. Update `PlayerEligibilityService.get_batter_positions` and
   `get_pitcher_positions` to read thresholds from `league.eligibility` instead of
   hardcoded values.
4. Update `fbm.toml` to include the eligibility section (optional — defaults match
   Yahoo).
5. Write tests: custom thresholds override defaults; missing section uses defaults;
   `carryover_seasons=0` disables multi-season logic; `carryover_seasons=2` looks back
   two years.

### Acceptance criteria

- `EligibilityRules` dataclass with sensible defaults matching Yahoo rules.
- Thresholds are parsed from TOML and flow through to the eligibility service.
- Missing `[eligibility]` section uses defaults (backward compatible).
- `carryover_seasons` controls how many prior seasons are included.
- All existing tests pass without config changes (defaults match phase 1 behavior).

## Ordering

Phase 1 is the priority — it fixes the immediate problem of too few positive-value
players. Phase 2 is a follow-up that makes the system flexible for non-Yahoo leagues.
Phase 1 has no external dependencies. Phase 2 depends on phase 1.
