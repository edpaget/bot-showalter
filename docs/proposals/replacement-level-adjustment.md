# Replacement-Level Adjustment for Cross-Pool Draft Rankings

## Goal

Add a replacement-level adjustment stage to the valuation pipeline so that batter and pitcher values are comparable on a single scale. Currently both the z-score and ml-ridge valuators score batters against batters and pitchers against pitchers independently, then dump both lists into `DraftState` with no cross-pool calibration. This causes pitchers to be systematically overranked relative to ADP — Logan Webb at rank 31 with ADP 41, Zack Wheeler at rank 12 with ADP 135, and nearly every pitcher showing a large positive Diff.

## Background

### Why the current system overvalues pitchers

1. **Independent normalization.** Z-score normalizes each pool to mean=0, std=1. With 2,070 batters and 1,302 pitchers in the projection pool, the right tail of the pitcher distribution produces higher z-scores per player because the pool is smaller and more homogeneous.

2. **No replacement-level baseline.** In a 12-team league with the default roster (12 batter slots + 13 pitcher slots per team), roughly 144 batters and 156 pitchers are rostered. The 145th batter is far less productive than the average batter, and the 157th pitcher is far less productive than the average pitcher — but these replacement costs are different in magnitude and are never subtracted.

3. **ADP encodes risk discounting.** Real drafters discount pitchers for injury volatility, inconsistency, and waiver-wire depth. Pitcher projections correlate with outcomes at ~0.46 vs ~0.62 for hitters. This risk premium is reflected in ADP but not captured by either valuation method.

### What the literature says

The FanGraphs Great Valuation System Test (2014) tested 13 systems across 50 leagues. SGP with correct denominators won (r=0.9697), but all top-10 systems clustered within 0.9615–0.9697. The key differentiator was not the core method (z-score vs SGP vs PVM) but whether the system applied replacement-level and positional-scarcity adjustments correctly.

The standard pipeline in the fantasy analytics community is:

```
Projections → Value units (z-score or SGP) → Replacement-level subtraction → Dollar conversion
```

Our pipeline currently stops after step 2.

## Current State

### What we have

- **Z-score valuator** (`valuation/zscore.py`): per-category z-scores with correct marginal-contribution handling for rate stats. No replacement level.
- **SGP valuator** (`valuation/sgp.py`): per-category SGP using externally-provided denominators. No replacement level.
- **ML Ridge valuator** (`valuation/ml_valuate.py`): learns a composite value from projections→ADP. Trained per-pool, so the cross-pool problem persists.
- **Keeper replacement logic** (`keeper/replacement.py`): `DraftPoolReplacementCalculator` simulates a greedy draft to find replacement values per keeper slot. This logic is conceptually close to what we need but is specific to keeper surplus calculation.
- **Position data flow**: `DraftState` already receives `player_positions: dict[tuple[str, str], tuple[str, ...]]` — a mapping from `(player_id, position_type)` to eligible positions. Position data is available from Yahoo or a CSV file.
- **Roster config**: `DEFAULT_ROSTER_CONFIG` defines slots per position (2C, 1 1B, 1 2B, 1 SS, 1 3B, 5 OF, 1 Util, 9 SP, 4 RP, 3 BN).
- **League settings**: `team_count`, `batting_categories`, `pitching_categories` all configurable.

### The gap

No component computes position-specific replacement levels from the projection pool, and no stage subtracts replacement-level value from raw valuations before they enter `DraftState`.

## Design

### Approach: Value Over Replacement Player (VORP)

Implement a post-valuation adjustment that:

1. Determines how many players are rostered at each position
2. Identifies the replacement-level player at each position
3. Subtracts replacement-level value from each player's raw value
4. Produces adjusted values where 0.0 = replacement level

This approach is method-agnostic — it works on top of z-score, SGP, or ml-ridge output. It operates on `list[PlayerValue]` and returns `list[PlayerValue]` with adjusted `total_value` and `category_values`.

### Why not dollar conversion?

Dollar conversion (mapping VORP to auction dollars) adds value for auction leagues but is unnecessary for snake-draft rankings. VORP alone solves the cross-pool ranking problem. Dollar conversion can be added later as a separate concern.

### Why not change the Valuator protocol?

The replacement-level adjustment requires position data, roster config, and league settings — none of which the `Valuator` protocol currently accepts. Rather than bloating the protocol, this is better modeled as a separate pipeline stage that runs after valuation.

## Implementation Plan

### Phase 1: Replacement-level calculator

**New file:** `src/fantasy_baseball_manager/valuation/replacement.py`

Core data types:

```python
@dataclass(frozen=True)
class PositionThreshold:
    position: str
    roster_spots: int          # teams * slots_at_position
    replacement_rank: int      # roster_spots + 1
    replacement_value: float   # avg total_value of players around the threshold

@dataclass(frozen=True)
class ReplacementConfig:
    team_count: int
    roster_config: RosterConfig
    smoothing_window: int = 5  # average this many players around threshold
```

Core function:

```python
def compute_replacement_levels(
    batter_values: list[PlayerValue],
    pitcher_values: list[PlayerValue],
    player_positions: dict[str, tuple[str, ...]],
    config: ReplacementConfig,
) -> dict[str, PositionThreshold]:
    """Compute replacement-level value for each roster position.

    For each position (C, 1B, 2B, SS, 3B, OF, SP, RP):
    1. Filter players eligible at that position
    2. Sort by total_value descending
    3. Identify the replacement threshold: teams * slots
    4. Average the values in a window around the threshold
    5. Return the replacement-level value

    Multi-position players are counted at their scarcest eligible position
    (C > SS > 2B > 3B > 1B > OF for batters, SP > RP for pitchers).
    """
```

Position assignment for multi-eligible players uses a scarcity priority to avoid double-counting a player at multiple positions. The priority order is: C, SS, 2B, 3B, 1B, OF (matching standard fantasy analysis conventions where catcher is scarcest and outfield is deepest).

### Phase 2: VORP adjustment function

```python
def apply_replacement_adjustment(
    player_values: list[PlayerValue],
    player_positions: dict[str, tuple[str, ...]],
    thresholds: dict[str, PositionThreshold],
) -> list[PlayerValue]:
    """Subtract replacement-level value from each player.

    Each player's adjusted total_value = raw total_value - replacement_value
    at their assigned position. Players below replacement get negative values
    (they will rank last and effectively be filtered out).

    For category-based valuators (z-score, SGP), also adjusts per-category
    values proportionally so category breakdowns remain meaningful.

    For composite valuators (ml-ridge) where category_values is empty,
    only total_value is adjusted.
    """
```

### Phase 3: Integrate into draft-rank CLI

**File:** `src/fantasy_baseball_manager/draft/cli.py`

After the valuator produces `all_values`, and before `DraftState` is constructed, insert the replacement-level adjustment:

```python
valuator = container.create_valuator(method)
# ... produce batting_result, pitching_result ...

# Apply replacement-level adjustment
from fantasy_baseball_manager.valuation.replacement import (
    ReplacementConfig,
    apply_replacement_adjustment,
    compute_replacement_levels,
)

repl_config = ReplacementConfig(
    team_count=league_settings.team_count,
    roster_config=roster_config,
)
thresholds = compute_replacement_levels(
    batter_values=batting_result.values,
    pitcher_values=pitching_result.values,
    player_positions=player_positions,
    config=repl_config,
)
all_values = apply_replacement_adjustment(
    player_values=all_values,
    player_positions=player_positions,
    thresholds=thresholds,
)
```

Add a `--no-replacement` flag (default: replacement adjustment ON) so users can compare raw vs adjusted rankings.

### Phase 4: Tests

**New file:** `tests/valuation/test_replacement.py`

Key test cases:

- `test_replacement_level_at_correct_rank`: for a position with N roster spots, the replacement player is at rank N+1
- `test_smoothing_window_averages_correctly`: the 5-player window around the threshold produces a stable value
- `test_multi_position_assigns_scarcest`: a SS/2B-eligible player is counted as SS, not 2B
- `test_vorp_subtracts_position_replacement`: after adjustment, the replacement-level player has value ~0.0
- `test_scarce_position_gets_larger_boost`: catchers receive a larger upward adjustment than outfielders
- `test_empty_category_values_adjusts_total_only`: ml-ridge output (no category breakdown) works correctly
- `test_no_replacement_flag_skips_adjustment`: CLI respects --no-replacement

### Phase 5: Update orchestration and draft-simulate

Update `build_projections_and_positions()` in `shared/orchestration.py` to also apply replacement-level adjustment (it currently hardcodes zscore with no adjustment). This ensures `draft-simulate` also benefits.

## Position Assignment Algorithm

Multi-eligible players must be assigned to exactly one position for replacement-level calculation. The assignment proceeds in scarcity order:

1. Sort positions by scarcity: C, SS, 2B, 3B, 1B, OF, SP, RP
2. For each position (in scarcity order):
   a. Collect all players eligible at this position who haven't been assigned yet
   b. Sort by total_value descending
   c. Assign the top `roster_spots` players to this position
   d. Mark assigned players as consumed
3. Remaining unassigned players go to Util/BN (these are below replacement at every position)

This greedy assignment maximizes the scarcity effect: the best SS/2B goes to SS (scarcer), pushing down the SS replacement level and boosting all shortstops.

## Expected Impact

Using the default 12-team roster (12 batters + 13 pitchers per team):

- **Batters rostered:** ~144 (12 × 12). The 145th batter has a z-score around -2 to -4. Elite batters (z-score ~30) get adjusted to ~32-34 VORP.
- **Pitchers rostered:** ~156 (13 × 12). The 157th pitcher has a z-score around -5 to -8. Elite pitchers (z-score ~20) get adjusted to ~25-28 VORP.
- **Net effect:** The batter-pitcher gap widens because the pitcher pool has a lower replacement level (more pitchers rostered relative to pool size). This pushes pitchers down in the combined rankings, better matching ADP.
- **Positional scarcity:** Catchers and shortstops get a boost (high replacement cost at those positions), while outfielders and first basemen are discounted (deep pools).

## Verification

1. Run `draft-rank --method zscore` before and after: compare Diff column — pitcher Diffs should move closer to 0
2. Compute Spearman rank correlation with ADP before and after: target improvement of 0.05+
3. Spot-check that catcher rankings improve (Cal Raleigh, Will Smith should rank higher relative to ADP)
4. Confirm `--no-replacement` produces identical output to current behavior

## Future Extensions

- **Dollar conversion:** Map VORP to auction dollars using a configurable hitter/pitcher budget split (default 67/33). Formula: `dollars = (player_vorp / pool_total_vorp) * pool_budget + $1`.
- **Risk discounting:** Apply a pitcher-specific multiplier (e.g., 0.85) to account for volatility. Could be learned from historical projection-vs-outcome variance.
- **SGP denominators:** If league-specific historical standings data becomes available, SGP with proper denominators would replace z-score as the default valuation. The replacement-level stage works identically on top of SGP output.
- **Iterative pool refinement:** Bootstrap the replacement level by simulating a greedy draft, recalculating replacement levels, and repeating until convergence. This handles edge cases where the initial position assignment affects the replacement threshold.
