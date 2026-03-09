# Opponent Draft Model Roadmap

Track opponent roster construction during a live draft and surface actionable intelligence: which positions opponents still need, when a position run is developing, and which players are at risk of being taken before your next pick. Rather than just showing what's been picked, this tool analyzes the draft flow and predicts what's coming.

This roadmap depends on: draft state engine (done), draft board service (done), ADP integration (done). It benefits from: tier generator (done), positional scarcity (done).

## Status

| Phase | Status |
|-------|--------|
| 1 — Opponent need tracking | done (2026-03-09) |
| 2 — Position run detection | not started |
| 3 — Threat prediction | not started |

## Phase 1: Opponent need tracking

Track each opponent's filled and unfilled roster slots and expose a "league needs" view showing which positions are still in demand across the league.

### Context

The draft state engine already records every team's roster via `team_rosters`. What's missing is analysis: which positions does each team still need, and how does aggregate league demand affect what's available for the user? For example, if 8 of 12 teams still need a catcher and there are only 10 rosterable catchers left, catcher scarcity is imminent.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/opponent_model.py`:
   - `TeamNeeds` frozen dataclass: `team_idx`, `team_name` (optional), `filled: dict[str, int]`, `unfilled: dict[str, int]`, `total_value` (sum of drafted player values).
   - `LeagueNeeds` frozen dataclass: `teams: list[TeamNeeds]`, `demand_by_position: dict[str, int]` (total unfilled slots across all teams), `supply_by_position: dict[str, int]` (available players at each position), `scarcity_ratio: dict[str, float]` (demand / supply — higher means scarcer).
2. Implement `compute_league_needs()` in `src/fantasy_baseball_manager/services/opponent_model.py`:
   - Takes the current `DraftState`, league settings (for roster slot definitions), and the available player pool.
   - Computes per-team needs from roster slots minus filled positions.
   - Computes supply from available pool, counting players at each position.
   - Returns `LeagueNeeds` with demand/supply ratios.
3. Write tests verifying correct need computation at different draft stages (early, mid, late).

### Acceptance criteria

- `compute_league_needs()` correctly identifies unfilled slots for each team.
- `scarcity_ratio` is highest for positions where demand approaches or exceeds supply.
- Works at any point in the draft (empty, partial, nearly complete).

## Phase 2: Position run detection

Detect when multiple teams draft the same position in consecutive picks ("position runs") and alert the user that supply at that position is rapidly shrinking.

### Context

Position runs are a common draft dynamic — one team takes a shortstop, then two more follow suit because they see the position drying up. Detecting runs early gives the user a chance to jump in before the position is depleted or to stay disciplined and wait for value elsewhere.

### Steps

1. Add `PositionRun` frozen dataclass to the domain: `position`, `picks` (list of recent picks at this position), `run_length`, `remaining_supply`, `urgency` ("critical" / "developing" / "none").
2. Implement `detect_position_runs()` in the opponent model service:
   - Scans the last N picks (configurable window, default 2 * num_teams) for clusters of same-position picks.
   - A run is "developing" when 2+ picks at the same position occur within a half-round window.
   - A run is "critical" when 3+ picks occur AND remaining supply at that position is < 1.5x the user's remaining need at that position.
   - Returns active runs sorted by urgency.
3. Write tests with crafted pick sequences that trigger and don't trigger run detection.

### Acceptance criteria

- Detects a run when 3 shortstops are taken in a 6-pick window in a 12-team league.
- Does not false-positive on spread-out picks at the same position.
- "Critical" urgency fires only when supply is genuinely thin relative to remaining demand.
- Works for both snake and auction formats (auction runs are based on recent time window rather than pick sequence).

## Phase 3: Threat prediction

Predict which players on the user's target list are likely to be taken before their next pick, based on opponent needs and ADP.

### Context

The recommendation engine suggests who to draft, but doesn't warn about who's about to disappear. If the user's top target at SS has ADP just above the next pick and three teams ahead still need a shortstop, that player is a high-risk target. This phase combines opponent needs with ADP to estimate threat levels.

### Steps

1. Add `ThreatAssessment` frozen dataclass: `player_id`, `player_name`, `position`, `value`, `adp`, `picks_until_user_next`, `teams_needing_position` (count of teams picking before user who need this position), `threat_level` ("safe" / "at-risk" / "likely-gone").
2. Implement `assess_threats()` in the opponent model service:
   - Takes the recommendation list (or top-N available), the current draft state, opponent needs, ADP data, and the user's next pick number.
   - For each recommended player, count how many teams picking between now and the user's next turn need that player's position.
   - Cross-reference with ADP: if a player's ADP falls within the pick range before the user's turn AND teams in that range need the position, threat level is "at-risk" or "likely-gone."
   - Sort by threat level descending so the user sees the most urgent targets first.
3. Integrate into the draft REPL:
   - Add a `threats` command that displays the threat assessment.
   - Optionally annotate the `best` command output with threat indicators (e.g., a warning icon next to at-risk players).
4. Write tests with known draft positions, ADP data, and opponent needs.

### Acceptance criteria

- Players with ADP in the danger zone and multiple teams needing the position are flagged "likely-gone."
- Players with ADP well beyond the user's next pick are flagged "safe."
- Threat assessment updates after each pick as opponent needs change.
- The `threats` command provides actionable output (who to reach for vs. who to wait on).

## Ordering

Phases are sequential: 1 -> 2 -> 3. Phase 1 (need tracking) is the foundation that phase 2 and 3 build on. Phase 2 (run detection) and phase 3 (threat prediction) are independent of each other but both require phase 1. Phase 3 is the highest-value feature for draft-day decisions.
