# Schedule / Matchup Analyzer Roadmap

For H2H category leagues, analyze the MLB schedule to identify players with favorable early-season matchups, off-day patterns, and streaming-friendly schedules. This tool helps optimize the first few weeks after the draft — when schedule edges are most exploitable because all teams have full rosters and streaming is less competitive. Also useful for identifying schedule-based tiebreakers between similarly valued players on draft day.

This roadmap depends on: league settings (done — `LeagueFormat.H2H_CATEGORIES` supported), projections (done). Requires: new MLB schedule data ingest.

## Phase 1: MLB Schedule Ingest

Ingest the MLB regular season schedule so the system knows which teams play on which dates, how many games per week, and opponent matchups.

### Context

No MLB schedule data currently exists in the system. Statcast data has game-level information but only after games are played — it can't provide future schedules. The MLB Stats API provides full season schedules, including spring training, regular season, and postseason, with game dates, opponents, and home/away status.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/schedule.py`:
   - `ScheduledGame` frozen dataclass: `game_date: str` (ISO date), `home_team: str`, `away_team: str`, `game_pk: int` (MLB game ID), `doubleheader: bool`.
   - `TeamWeek` frozen dataclass: `team: str`, `week_number: int`, `start_date: str`, `end_date: str`, `games: int`, `opponents: list[str]`, `home_games: int`, `away_games: int`.
   - `SeasonSchedule` frozen dataclass: `season: int`, `games: list[ScheduledGame]`, `opening_day: str`, `weeks: dict[str, list[TeamWeek]]` (team -> weekly breakdown).
2. Build MLB schedule fetcher in `src/fantasy_baseball_manager/ingest/sources/mlb_schedule.py`:
   - Fetch from MLB Stats API: `https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&gameType=R`.
   - Parse JSON response into `list[ScheduledGame]`.
   - Handle doubleheaders (count as 2 games in weekly totals).
3. Build `aggregate_weekly_schedule()` in `src/fantasy_baseball_manager/services/schedule.py`:
   - Accepts `games: list[ScheduledGame]`, `season: int`, `week_start_day: str = "monday"`.
   - Groups games into fantasy-relevant weeks (Monday-Sunday by default).
   - Produces `TeamWeek` records for each team-week combination.
4. Add `ScheduledGame` persistence in a new `schedule` table and repo following existing repository patterns.
5. Add `fbm ingest schedule --season <year>` CLI command.
6. Write tests for schedule parsing, weekly aggregation, and doubleheader handling.

### Acceptance criteria

- `fbm ingest schedule --season 2026` fetches and stores the full MLB regular season schedule.
- Weekly aggregation correctly groups games into Monday-Sunday fantasy weeks.
- Doubleheaders are counted as 2 games in weekly totals.
- Each team has ~162 games across ~26 weeks.

## Phase 2: Team Strength Ratings

Compute opponent strength ratings per team so the schedule analyzer can distinguish between favorable and tough matchups.

### Context

Playing 7 games in a week is better than 5, but 7 games against the Dodgers pitching staff is worse than 5 games against weaker opponents. Team strength ratings (based on prior-season performance or preseason projections) add a quality dimension to schedule analysis.

### Steps

1. Define types in `domain/schedule.py`:
   - `TeamStrength` frozen dataclass: `team: str`, `season: int`, `batting_strength: float` (z-score vs. league avg), `pitching_strength: float`, `overall_strength: float`, `source: str` ("prior_season" or "projected").
   - `WeekMatchupQuality` frozen dataclass: `team: str`, `week_number: int`, `games: int`, `avg_opponent_pitching: float` (for batters — lower = easier), `avg_opponent_batting: float` (for pitchers — lower = easier), `schedule_score: float` (composite: games weighted by opponent weakness).
2. Build `compute_team_strengths()` in `services/schedule.py`:
   - Accepts prior-season team-level stats (aggregate from player batting/pitching stats by team).
   - Computes z-scores for batting (runs, OBP, SLG) and pitching (ERA, WHIP, K/9) relative to league average.
   - Returns `list[TeamStrength]`.
3. Build `compute_weekly_matchup_quality()` in `services/schedule.py`:
   - Accepts `weekly_schedule: list[TeamWeek]`, `strengths: list[TeamStrength]`.
   - For each team-week, computes average opponent strength (pitching strength for batter matchups, batting strength for pitcher matchups).
   - `schedule_score = games * (1 - avg_opponent_strength)` — more games against weaker opponents = higher score.
   - Returns `list[WeekMatchupQuality]`.
4. Write tests verifying that a team playing 7 games against weak pitching scores higher than one playing 5 games against strong pitching.

### Acceptance criteria

- Team strength ratings are computed from prior-season aggregate stats.
- Weekly matchup quality correctly favors more games against weaker opponents.
- `schedule_score` is comparable across teams and weeks (normalized).

## Phase 3: Player Schedule Report

Map team schedules to individual players to produce player-level schedule advantage reports for early-season fantasy weeks.

### Context

Drafters care about players, not teams. This phase maps team-level schedule quality to individual players, producing a report that says "Player X has 15 games in the first 2 weeks against bottom-10 pitching staffs." This becomes a tiebreaker on draft day and a streaming target identifier.

### Steps

1. Define types in `domain/schedule.py`:
   - `PlayerScheduleEdge` frozen dataclass: `player_id: int`, `player_name: str`, `team: str`, `player_type: str`, `position: str`, `weeks_analyzed: int`, `total_games: int`, `avg_schedule_score: float`, `best_week: int` (week number with highest schedule score), `worst_week: int`, `projected_value: float` (from projections), `schedule_adjusted_value: float`.
2. Build `compute_player_schedule_edges()` in `services/schedule.py`:
   - Accepts `weekly_quality: list[WeekMatchupQuality]`, `roster_stints: list[RosterStint]` (to map players to teams), `projections: list[Projection]`, `weeks_to_analyze: int = 4` (first N fantasy weeks).
   - Maps each player to their team's weekly matchup quality.
   - Computes `schedule_adjusted_value = projected_value * (1 + schedule_bonus)` where `schedule_bonus` is a small multiplier (e.g., 0-5%) based on schedule score relative to league average.
   - Returns `list[PlayerScheduleEdge]` sorted by `schedule_adjusted_value` descending.
3. Build `find_streaming_targets()` variant:
   - Filters to pitchers with moderate projections but excellent single-week matchups.
   - Identifies SP with multiple starts in a favorable week.
   - Returns a separate ranked list for streaming decisions.
4. Write tests with synthetic schedules and projections.

### Acceptance criteria

- Players on teams with favorable early-season schedules are boosted.
- Schedule adjustment is small enough to be a tiebreaker, not a dominant factor (max ~5% value swing).
- Streaming targets correctly identify pitchers with favorable single-week matchups.
- Only the first N weeks are analyzed (configurable), not the full season.

## Phase 4: CLI Commands

Expose schedule analysis through the CLI.

### Steps

1. Add `fbm report schedule-edges --season <year> --weeks <n> --system <system>`:
   - Prints players with the biggest schedule advantages in the first N weeks.
   - Columns: player, team, position, games, avg schedule score, projected value, schedule-adjusted value.
   - Supports `--player-type batter|pitcher`, `--position <pos>`, `--top <n>`.
2. Add `fbm report streaming-targets --season <year> --week <n>`:
   - Prints best streaming pitcher targets for a specific week.
   - Columns: pitcher, team, starts, opponents, opponent batting strength, schedule score.
3. Add `fbm report team-schedule --season <year> --team <abbrev> --weeks <n>`:
   - Prints a single team's weekly schedule breakdown with matchup quality.
4. Register commands in `cli/app.py` under the `report` group.

### Acceptance criteria

- `fbm report schedule-edges` identifies players with early-season schedule advantages.
- `fbm report streaming-targets` highlights pitchers with favorable single-week starts.
- `fbm report team-schedule` shows a week-by-week breakdown for one team.
- All commands require schedule data to be ingested first and error clearly if missing.

## Ordering

Phase 1 is independent and must come first — it provides the raw schedule data. Phase 2 depends on phase 1 (needs games to rate matchups). Phase 3 depends on phase 2 (needs matchup quality) and existing projections/roster stints. Phase 4 depends on all prior phases. This roadmap is self-contained — no other roadmap depends on it, and it has no hard dependencies on planned-but-unbuilt features. Best implemented close to draft season when the MLB schedule for the upcoming year is published.
