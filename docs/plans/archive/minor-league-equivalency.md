# Minor League Equivalency (MLE) — Roadmap

## Goal

Translate minor league batting statistics into MLB-equivalent lines so the projection system can produce meaningful forecasts for players with limited or no major league history. A 22-year-old with 500 PA at AA should produce a useful projection — today the system ignores him entirely because Marcel requires MLB data.

---

## Problem Statement

The current projection pipeline (Marcel, Statcast GBM, ensemble) operates exclusively on major league statistics. This means:

- **Prospects and call-ups have no projection.** A top-100 prospect with three years of minor league data gets treated the same as a replacement-level unknown — both produce nothing.
- **Players with thin MLB samples get heavily regressed.** A player with 80 MLB PA gets regressed almost entirely to league average. His 400 PA at AAA the same year could meaningfully improve the estimate, but it's discarded.
- **No pre-call-up signal.** Fantasy managers need projections for players *before* they're promoted. The system can't provide this without MLE capability.

Minor league equivalencies are the standard approach to this problem. The core idea: translate minor league stats through park, league, and competition-level adjustments to produce what those stats would look like in an MLB context, then feed the translated line into the existing projection machinery alongside (or instead of) MLB data.

---

## Design

### Translation Approach: Component-Level Rate Translation

Rather than translating a single summary stat (like OPS), the system translates individual rate components separately. Each component has a different translation factor and a different reliability:

| Component | Translates | Reliability |
|-----------|-----------|-------------|
| K% | Well — stabilizes in ~60 PA, r ≈ 0.77 MiLB→MLB | High |
| BB% | Well from AA+ — stabilizes in ~120 PA, r ≈ 0.72 | High (upper levels) |
| ISO | Moderately — meaningful predictor at every level | Medium |
| HR rate | Moderately — park-sensitive but r ≈ 0.59 | Medium |
| BABIP | Poorly — requires ~820 BIP to stabilize, inflated norms at every MiLB level | Low |
| 3B rate | Poorly — park and outfield defense dependent | Low |

This component-level approach means we can regress each component independently (heavy regression on BABIP, light on K%) rather than applying uniform regression to a composite stat.

### Level Quality Factors

Each minor league level has an empirically derived quality factor relative to MLB (1.00):

| Level | Approximate Factor | Interpretation |
|-------|-------------------|----------------|
| AAA | 0.78–0.82 | ~20% performance reduction |
| AA | 0.65–0.70 | ~30–35% reduction |
| High-A | 0.55–0.60 | ~40–45% reduction |
| Low-A | 0.40–0.45 | ~55–60% reduction |
| Rookie | 0.30–0.35 | ~65–70% reduction |

The classic Szymborski/James formula uses `m = PL × 0.82` for AAA, where PL is the park/league run-environment ratio. For lower levels, factors are chained. These factors should be recalibrated periodically (rule changes like the automated strike zone in AAA shift the numbers).

### Age-for-Level Adjustment

Age-for-level is as important as the stats themselves. A 20-year-old hitting well at AA is a fundamentally different prospect than a 25-year-old with the same numbers. The system incorporates age as a first-class adjustment, not an afterthought.

Optimal age benchmarks: Low-A = 19, High-A = 20, AA = 21, AAA = 22. Players younger than benchmark get favorable adjustments; older players are penalized.

### Blending MLE and MLB Data

When a player has both minor and major league stats, the system blends them with a reliability discount on MLE data. Research suggests MLE PA carry roughly 50–60% the information value of actual MLB PA:

```
blended_rate = (MLB_PA × mlb_rate + MLE_PA × discount × mle_rate + regression_PA × lg_avg)
             / (MLB_PA + MLE_PA × discount + regression_PA)
```

This naturally handles the full spectrum: a prospect with 0 MLB PA uses only MLE data (heavily regressed); a sophomore with 200 MLB PA and 300 MLE PA blends both; an established player with 2000+ MLB PA sees negligible MLE influence.

### Output

MLE translations produce `BattingStats`-compatible records tagged with a distinct system identifier (e.g., `system="mle"`, `version="v1"`). These flow into the existing projection pipeline:

- Marcel can consume MLE-translated seasons as historical input alongside real MLB seasons.
- The ensemble can weight MLE-based projections as a component.
- Distributions are computed via the existing `samples_to_distribution` infrastructure.

---

## Phases

### Phase 1 — Minor League Stats Ingest

Bring minor league batting statistics into the database. This is the blocking prerequisite for everything else.

#### 1a. Domain type

**File:** `domain/minor_league_stats.py` (new)

```python
@dataclass(frozen=True)
class MinorLeagueBattingStats:
    player_id: int
    season: int
    level: str            # "AAA", "AA", "A+", "A", "ROK"
    league: str           # "International League", "Pacific Coast League", etc.
    team: str             # MiLB team name
    org: str              # Parent MLB organization
    g: int
    pa: int
    ab: int
    h: int
    doubles: int
    triples: int
    hr: int
    r: int
    rbi: int
    bb: int
    so: int
    sb: int
    cs: int
    hbp: int
    sf: int
    sh: int
    avg: float
    obp: float
    slg: float
    age: float            # Age during the season
    id: int | None = None
    loaded_at: str | None = None
```

#### 1b. Database migration

**File:** `db/migrations/NNN_minor_league_batting_stats.sql` (new)

```sql
CREATE TABLE minor_league_batting_stats (
    id          INTEGER PRIMARY KEY,
    player_id   INTEGER NOT NULL REFERENCES player(id),
    season      INTEGER NOT NULL,
    level       TEXT NOT NULL,
    league      TEXT NOT NULL,
    team        TEXT NOT NULL,
    org         TEXT NOT NULL,
    g           INTEGER NOT NULL,
    pa          INTEGER NOT NULL,
    ab          INTEGER NOT NULL,
    h           INTEGER NOT NULL,
    doubles     INTEGER NOT NULL,
    triples     INTEGER NOT NULL,
    hr          INTEGER NOT NULL,
    r           INTEGER NOT NULL,
    rbi         INTEGER NOT NULL,
    bb          INTEGER NOT NULL,
    so          INTEGER NOT NULL,
    sb          INTEGER NOT NULL,
    cs          INTEGER NOT NULL,
    hbp         INTEGER NOT NULL,
    sf          INTEGER NOT NULL,
    sh          INTEGER NOT NULL,
    avg         REAL NOT NULL,
    obp         REAL NOT NULL,
    slg         REAL NOT NULL,
    age         REAL NOT NULL,
    loaded_at   TEXT,
    UNIQUE(player_id, season, level, team)
);
```

#### 1c. Repository

**File:** `repos/minor_league_stats_repo.py` (new)

Protocol `MinorLeagueBattingRepo`:
- `upsert(stats: MinorLeagueBattingStats, conn) -> MinorLeagueBattingStats`
- `get_by_player(player_id: int, conn) -> list[MinorLeagueBattingStats]`
- `get_by_player_season(player_id: int, season: int, conn) -> list[MinorLeagueBattingStats]`
- `get_by_season_level(season: int, level: str, conn) -> list[MinorLeagueBattingStats]`

Implement as `SqliteMinorLeagueBattingRepo`.

#### 1d. Data source

**File:** `ingest/milb_batting_source.py` (new)

Fetch minor league batting stats via pybaseball or the MLB Stats API. The MLB Stats API endpoint `https://statsapi.mlb.com/api/v1/stats?group=hitting&type=season&sportId=N` covers each level (sportId: 11=AAA, 12=AA, 13=A+, 14=A, etc.).

Alternatively, FanGraphs exposes minor league leaderboards via pybaseball's `fg_milb_batting_data()` function, which provides more advanced metrics (wRC+, K%, BB%) out of the box.

Evaluate both sources; prefer whichever provides more complete data with reliable player ID matching (mlbam_id linkage is critical).

**CLI:** `fbm ingest milb-batting [--season YEAR] [--level LEVEL]`

#### 1e. Tests

- Round-trip: insert a `MinorLeagueBattingStats`, retrieve by player, verify fields.
- Upsert idempotency: same player/season/level/team updates rather than duplicates.
- `get_by_season_level` returns all players at a given level for a season.
- Column mapper correctly handles edge cases (players who played at multiple levels in one season produce separate rows).
- Integration test: mock data source → loader → database → query.

---

### Phase 2 — League Environment Context

Build the reference data needed for park and league adjustments.

#### 2a. League-season run environment

**File:** `domain/league_environment.py` (new)

```python
@dataclass(frozen=True)
class LeagueEnvironment:
    league: str           # "International League", "MLB", etc.
    season: int
    level: str            # "AAA", "AA", "A+", "A", "ROK", "MLB"
    runs_per_game: float
    avg: float            # League batting average
    obp: float
    slg: float
    k_pct: float
    bb_pct: float
    hr_per_pa: float
    babip: float
```

Compute from aggregate minor league stats per league per season (from the data ingested in Phase 1). Store in a `league_environment` table.

#### 2b. Level quality factors

**File:** `domain/level_factor.py` (new)

```python
@dataclass(frozen=True)
class LevelFactor:
    level: str
    season: int
    factor: float         # Quality relative to MLB (e.g., 0.80 for AAA)
    k_factor: float       # Component-specific factor for K%
    bb_factor: float      # Component-specific factor for BB%
    iso_factor: float     # Component-specific factor for ISO
    babip_factor: float   # Component-specific factor for BABIP
```

Initial values are seeded from published research (Szymborski/Davenport). A future phase could empirically recalibrate by tracking players who transition between levels mid-season.

#### 2c. Tests

- League environment computed from known aggregate data matches expected values.
- Level factors produce sensible relative ordering (AAA > AA > A+ > A > ROK).
- Component-specific factors differ from each other (K factor ≠ BABIP factor).

---

### Phase 3 — MLE Translation Engine

The core computation: translate a minor league batting line into an MLB-equivalent line.

#### 3a. Translation function

**File:** `models/mle/engine.py` (new)

```python
def translate_batting_line(
    stats: MinorLeagueBattingStats,
    league_env: LeagueEnvironment,
    mlb_env: LeagueEnvironment,
    level_factor: LevelFactor,
    config: MLEConfig,
) -> TranslatedBattingLine:
    """Translate a minor league batting line to MLB-equivalent."""
```

**Steps:**

1. **Park/league ratio:** `PL = milb_rpg / mlb_rpg`
2. **Competition factor:** `m = PL × level_factor.factor`
3. **Rate factor:** `M = sqrt(m)`
4. **Component translation:**
   - K% → `milb_k_pct × level_factor.k_factor` (with configurable multiplier, default ~1.05–1.22 depending on experience)
   - BB% → `milb_bb_pct × level_factor.bb_factor`
   - ISO → `milb_iso × level_factor.iso_factor`
   - BABIP → regressed toward MLB league average, weighted by PA and stabilization rate (~820 BIP)
   - HR → `milb_hr × m × park_mult`
   - 2B → `milb_2b × M × park_mult`
   - 3B → `milb_3b × m × 0.85 × park_mult`
5. **Reconstruct counting stats** from translated rates and original outs.
6. **Derive AVG, OBP, SLG** from translated components.

#### 3b. Output type

```python
@dataclass(frozen=True)
class TranslatedBattingLine:
    player_id: int
    season: int
    source_level: str
    pa: int               # Same PA as original (translation preserves opportunities)
    ab: int               # Reconstructed from translated H + original outs
    h: int
    doubles: int
    triples: int
    hr: int
    bb: int
    so: int
    hbp: int
    sf: int
    avg: float
    obp: float
    slg: float
    k_pct: float
    bb_pct: float
    iso: float
    babip: float
```

#### 3c. Configuration

```python
@dataclass(frozen=True)
class MLEConfig:
    babip_regression_weight: float = 0.5    # How much to regress BABIP toward league avg
    k_experience_factor: float = 1.15       # K% multiplier for first exposure to level
    min_pa: int = 100                       # Minimum PA to translate
    discount_factor: float = 0.55           # MLE PA reliability vs MLB PA (for blending)
```

#### 3d. Tests

- Known input produces expected translated line (hand-calculated test case).
- BABIP regression pulls toward MLB average — a .350 MiLB BABIP doesn't translate to .350.
- K% increases after translation (strikeouts go up at MLB).
- ISO decreases after translation (power is discounted).
- Edge cases: player with 0 HR, player with very high/low BABIP.
- Level ordering: same raw line translated from AAA produces better output than from AA.

---

### Phase 4 — Age-for-Level Adjustment

Add age context to translated lines, adjusting projections for how old a player is relative to his competition level.

#### 4a. Age adjustment function

**File:** `models/mle/age_adjustment.py` (new)

```python
def compute_age_adjustment(
    age: float,
    level: str,
    config: AgeAdjustmentConfig,
) -> float:
    """Return a multiplier (>1.0 favorable, <1.0 unfavorable) based on age-for-level."""
```

Uses the optimal age benchmarks (Low-A=19, High-A=20, AA=21, AAA=22). Each year younger than benchmark adds a bonus; each year older subtracts. The adjustment is asymmetric — being young is more informative than being old (a 19-year-old at AA is rare and strongly positive; a 26-year-old at AA is common and only mildly negative).

Also incorporates a development projection: younger players are expected to improve, so their translated line is adjusted upward to reflect projected growth to peak age (~26–27), using a configurable per-year improvement rate.

#### 4b. Wire into translation

The age adjustment multiplies the translated rate stats (or a subset — ISO and power-related stats may get more age adjustment than K%, which is already stable by the late minors).

#### 4c. Tests

- A 20-year-old at AA gets a higher adjustment than a 24-year-old at AA.
- The adjustment is bounded (no player gets a 2x multiplier).
- Adjustment at optimal age for level ≈ 1.0 (neutral).
- Younger-than-benchmark adjustments are larger in magnitude than older-than-benchmark.

---

### Phase 5 — MLE Projection Model

Wire the translation engine into the model registry so it integrates with the rest of the projection pipeline.

#### 5a. Model class

**File:** `models/mle/model.py` (new)

```python
@register("mle")
class MLEModel:
    name = "mle"
    supported_operations = frozenset({"prepare", "predict"})
```

Implements `Preparable` and `Predictable`. No training step — the translation is formula-based, not learned. Configuration (level factors, regression weights) is declarative.

#### 5b. Prepare

Materializes the feature set: for each player with minor league data in the relevant seasons, pull their `MinorLeagueBattingStats`, the corresponding `LeagueEnvironment`, and the `LevelFactor`. Players with stats at multiple levels in one season have each level translated separately.

#### 5c. Predict

For each player:
1. Translate each season × level line via the engine (Phase 3).
2. Apply age adjustment (Phase 4).
3. If multiple levels in one season, combine by PA-weighting the translated lines.
4. If multiple seasons, apply recency weights (heavier on recent seasons, matching Marcel's 5/4/3 or similar).
5. Regress the weighted-average translated line toward MLB league average, with regression amount determined by total effective PA (`MLE_PA × discount_factor`).
6. Output a `Projection` domain object with `system="mle"`.

#### 5d. Tests

- End-to-end: seed MiLB data → prepare → predict → verify projection is stored.
- Multi-level season: player with 200 PA at AA and 300 PA at AAA produces a blended line.
- Multi-season weighting: more recent seasons have higher weight.
- Regression: a player with 200 MLE PA is regressed more than one with 600 MLE PA.
- Output conforms to `Projection` schema and is readable by downstream models.

---

### Phase 6 — Blending MLE with MLB Data

Enable the projection pipeline to combine MLE translations with actual MLB stats for players who have both.

#### 6a. Blending function

**File:** `models/mle/blend.py` (new)

```python
def blend_mle_with_mlb(
    mlb_stats: list[BattingStats],
    mle_stats: list[TranslatedBattingLine],
    config: BlendConfig,
) -> BlendedStatLine:
    """Reliability-weighted blend of MLB and MLE data."""
```

For each rate component (K%, BB%, ISO, BABIP, etc.):
- Weight MLB PA at face value.
- Weight MLE PA at `discount_factor` (default 0.55).
- Add regression PA pushing toward MLB league average.
- The regression amount per component uses stat-specific stabilization rates (60 PA for K%, 820 BIP for BABIP, etc.).

#### 6b. Marcel integration

**File:** `models/marcel/engine.py`

Modify the Marcel input pipeline to accept blended stat lines for players with MLE data. When `MLEModel` has produced projections for a player, Marcel treats the blended line as an additional historical season (or as a replacement for missing MLB seasons).

The key design choice: MLE-translated seasons are tagged so Marcel can apply different weights to them versus real MLB seasons. A simple approach: MLE seasons receive a weight multiplier of `discount_factor` relative to MLB seasons in the same lag position.

#### 6c. Tests

- Pure MLE player (0 MLB PA): projection is entirely MLE-derived, heavily regressed.
- Mixed player (200 MLB PA + 400 MLE PA): blend appropriately weights both.
- Established player (2000+ MLB PA): MLE data has negligible effect.
- Blended K% uses K%-specific stabilization; blended BABIP uses BABIP-specific stabilization.
- Marcel with MLE input produces different (better) projections for call-ups than Marcel without.

---

## Phase Order

```
Phase 1 (MiLB stats ingest)
  ↓
Phase 2 (league environment + level factors)
  ↓
Phase 3 (translation engine)
  ↓
Phase 4 (age adjustment)
  ↓
Phase 5 (MLE model)
  ↓
Phase 6 (blending with MLB + Marcel integration)
```

All phases are sequential — each depends on the prior. Phase 2 depends on Phase 1 data being available. Phase 3 requires Phase 2's environment context. Phases 4 and 5 build on Phase 3's core engine. Phase 6 ties everything into the existing system.

---

## Out of Scope

- **Pitcher MLEs.** Pitcher translations are less reliable than hitter translations (defense quality variance, workload management, FIP vs ERA noise). Adding pitcher MLE support is a natural extension but a separate effort.
- **Comparable player matching.** KNN/similarity-based approaches (a la PECOTA) for deriving aging curves and outcome distributions from historical player cohorts. Valuable but adds significant complexity — better as a follow-on after the base MLE system is working.
- **Empirical level factor recalibration.** Tracking mid-season promotions/demotions to empirically derive level factors from our own data rather than using published values. Requires multiple seasons of data and careful handling of selection bias.
- **Statcast MiLB data.** Exit velocity, barrel rate, etc. from AAA Statcast tracking. Promising for improving translations but requires a separate ingest pipeline and only covers Triple-A publicly.
- **International leagues.** NPB, KBO, and independent league translations use the same conceptual framework but require separate quality factors and data sources.
- **Minor league park factors.** Individual park adjustments within minor leagues. Available data is noisy due to small samples and frequent league reorganization. The initial implementation uses league-level adjustments only.

---

## Open Questions

- **Data source selection.** The MLB Stats API provides raw counting stats with reliable mlbam_id linkage. FanGraphs (via pybaseball `fg_milb_batting_data()`) provides richer metrics (wRC+, K%, BB%) but player ID matching may require the Chadwick register. Need to evaluate which source is more complete and reliable.
- **Multi-level season handling.** When a player plays at two levels in one season, should we translate each stint separately and PA-weight, or combine the raw stats first? Separate translation is more principled (different level factors) but requires stint-level granularity.
- **Level factor granularity.** Should factors be per-level (AAA, AA, etc.) or per-league (International League, Pacific Coast League)? Per-league is more accurate (Davenport's approach) but requires more reference data. Start per-level, refine later.
- **Rule change recalibration.** The automated strike zone in AAA (since 2023) shifted K% and BB% significantly. Should the system maintain era-specific level factors, or is a single factor per level sufficient?
