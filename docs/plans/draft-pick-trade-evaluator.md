# Draft Pick Trade Evaluator Roadmap

Evaluate draft pick trades by estimating the surplus value of each pick slot. In leagues that allow trading draft picks before or during the draft, this tool answers questions like "is pick 15 + pick 60 worth pick 5 + pick 85?" by mapping pick numbers to expected player value using ADP-based position curves. Supports both snake and auction formats.

This roadmap depends on: ADP integration (done), valuations (done), draft board service (done).

## Status

| Phase | Status |
|-------|--------|
| 1 — Pick value curves | not started |
| 2 — Trade evaluation engine | not started |
| 3 — Multi-round cascade analysis | not started |
| 4 — CLI commands | not started |

## Phase 1: Pick Value Curves

Build the foundational model that maps draft pick numbers to expected player value, producing smooth value curves from historical ADP-to-valuation data.

### Context

The value of a draft pick depends on who you're likely to draft with it. By joining ADP data (what pick a player typically goes at) with ZAR valuations (what that player is worth), we can build a curve of expected value per pick number. This curve is the basis for all pick trade math.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/pick_value.py`:
   - `PickValue` frozen dataclass: `pick: int`, `expected_value: float`, `player_name: str | None` (the player at that ADP), `confidence: str` (high/medium/low based on ADP sample size).
   - `PickValueCurve` frozen dataclass: `season: int`, `provider: str`, `system: str`, `picks: list[PickValue]`, `total_picks: int`.
2. Build `compute_pick_value_curve()` in `src/fantasy_baseball_manager/services/pick_value.py`:
   - Accepts `adp: list[ADP]`, `valuations: list[Valuation]`, `league: LeagueSettings`.
   - Joins ADP records to valuations by `player_id`.
   - For each pick number 1..N, assigns the expected value of the player at that ADP slot.
   - Applies LOESS or rolling-average smoothing to handle gaps and noise in the raw mapping.
   - Returns `PickValueCurve`.
3. Write tests verifying the curve is monotonically non-increasing (higher picks have higher or equal value) and handles missing ADP-valuation joins gracefully.

### Acceptance criteria

- `compute_pick_value_curve()` returns a smooth, monotonically non-increasing value curve.
- Picks with no ADP-valuation join are interpolated from neighbors.
- The curve covers at least picks 1 through `teams * (roster_batters + roster_pitchers)`.

## Phase 2: Trade Evaluation Engine

Use pick value curves to evaluate trades by comparing the total expected value of picks given vs. picks received.

### Context

With a pick value curve in hand, evaluating a trade is straightforward: sum the expected values on each side and compare. But a useful evaluator also accounts for positional context — trading down from pick 5 to pick 15 costs more if the top-5 has elite shortstops and your roster needs one.

### Steps

1. Define trade types in `domain/pick_value.py`:
   - `PickTrade` frozen dataclass: `gives: list[int]` (pick numbers given away), `receives: list[int]` (pick numbers received).
   - `TradeEvaluation` frozen dataclass: `trade: PickTrade`, `gives_value: float`, `receives_value: float`, `net_value: float`, `gives_detail: list[PickValue]`, `receives_detail: list[PickValue]`, `recommendation: str` ("accept" / "reject" / "even").
2. Build `evaluate_pick_trade()` in `services/pick_value.py`:
   - Accepts `trade: PickTrade`, `curve: PickValueCurve`, `threshold: float = 1.0` (minimum net value to recommend accept).
   - Sums expected value for each side using the curve.
   - Returns `TradeEvaluation` with net value and recommendation.
3. Build `evaluate_pick_trade_with_context()` variant that also accepts the user's current roster and `DraftBoard`:
   - Estimates which player the user would likely draft at each pick (based on positional need + value).
   - Provides player-specific value estimates instead of curve averages.
4. Write tests for symmetric trades (same picks = net zero), obvious wins, and obvious losses.

### Acceptance criteria

- `evaluate_pick_trade()` correctly identifies that trading pick 1 for picks 20+21 is a loss when the curve is steep at the top.
- Context-aware evaluation changes the recommendation when positional need makes a lower pick more valuable.
- `net_value` is positive when receiving side has more total expected value.

## Phase 3: Multi-Round Cascade Analysis

Analyze how a pick trade affects the entire draft, not just the traded picks — because trading up in round 1 means trading down somewhere else.

### Context

Draft pick trades don't happen in isolation. If you trade pick 15 for pick 5, your opponent now picks at 15 instead of 5. This changes the available player pool at every subsequent pick. Cascade analysis simulates the downstream effects using the mock draft engine (if available) or a simplified greedy model.

### Steps

1. Build `cascade_analysis()` in `services/pick_value.py`:
   - Accepts `trade: PickTrade`, `board: DraftBoard`, `league: LeagueSettings`, `user_team_idx: int`.
   - Simulates two drafts: one with the original pick order, one with the traded pick order.
   - Uses a greedy best-value strategy for all teams.
   - Compares the user's roster quality (total value, category balance) between the two scenarios.
   - Returns `CascadeResult` with before/after roster values and the marginal impact.
2. If mock draft simulator (separate roadmap) is available, use `run_batch_simulation()` to run N simulations of each scenario for statistical robustness.
3. Write tests with small leagues (4-team, 5-round) to verify cascade picks differ between scenarios.

### Acceptance criteria

- `cascade_analysis()` shows different rosters in traded vs. non-traded scenarios.
- Total roster value difference is consistent with the `TradeEvaluation.net_value` from phase 2 (directionally, not exactly).
- Works without mock draft simulator (falls back to greedy single-simulation).

## Phase 4: CLI Commands

Expose pick trade evaluation through the CLI.

### Steps

1. Add `fbm draft pick-values --season <year> --system <system> --provider <adp-provider>`:
   - Prints the pick value curve as a table (pick number, expected value, likely player).
2. Add `fbm draft trade-picks --gives <picks> --receives <picks> --season <year> --system <system>`:
   - Prints trade evaluation with net value and recommendation.
   - Example: `fbm draft trade-picks --gives 5,85 --receives 15,60 --season 2026 --system zar`.
3. Add `--cascade` flag to `trade-picks` for cascade analysis (phase 3).
4. Register under the `draft` typer group in `cli/app.py`.

### Acceptance criteria

- `fbm draft pick-values` prints a clean table of pick values.
- `fbm draft trade-picks` prints gives/receives detail and a clear recommendation.
- `--cascade` flag triggers cascade analysis and shows before/after roster comparison.

## Ordering

Phase 1 is independent and can start immediately. Phase 2 depends on phase 1 (needs the curve). Phase 3 depends on phase 2 and benefits from the mock draft simulator roadmap but can work without it. Phase 4 depends on all prior phases. This roadmap is relatively self-contained — no external roadmap is a hard blocker.
