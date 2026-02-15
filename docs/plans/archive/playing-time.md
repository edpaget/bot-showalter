# Playing Time Projections — Roadmap

## Goal

Replace Marcel's mechanical playing time formula with a model that incorporates injury history, age-specific availability curves, and performance-based opportunity allocation. Marcel's `0.5*PA₁ + 0.1*PA₂ + 200` baseline explains roughly 8% of playing time variance. Published research shows a five-variable regression (age, prior PA, prior WAR, starter status, prior IL days) reaches 74%. The infrastructure and most of the data already exist in this system — the primary gap is injury history.

---

## Problem Statement

Playing time is the highest-leverage input in a fantasy projection. A player projected for a .280 AVG matters very differently at 650 PA versus 350 PA. Yet playing time is the weakest part of every public projection system, and the current Marcel implementation is especially crude:

- **No injury signal.** A player who spent half of last season on the IL gets the same baseline as one who played 162 games. Prior IL days are among the strongest predictors of future playing time loss.
- **Flat age treatment.** Marcel applies a single aging curve to rates but does not model the separate, steeper decline in games played with age. Players over 30 lose playing time faster than they lose skill.
- **No performance feedback.** A player who posted 5 WAR is far more likely to get a full-time role than one who posted 0.5 WAR. Marcel ignores this entirely.
- **Point estimate only.** Playing time has high variance — more than rate stats. A player recovering from a major injury might play 50 games or 150 games. The current system produces a single number with no indication of the uncertainty range.

---

## Design

### Model Architecture: Two-Stage Regression

The playing time model separates two questions:

1. **How many games will the player be available for?** (Availability model — driven by injury history, age, and position.)
2. **Given availability, how much playing time will the player receive?** (Opportunity model — driven by performance, roster competition, and starter status.)

Projected playing time = availability × opportunity share.

This decomposition is useful because the two components have different drivers, different data sources, and different error profiles. Injuries are high-variance discrete events; opportunity allocation is a smoother function of relative skill.

For the initial implementation, we combine both into a single regression (the Cartwright approach), since the data requirements are lighter and it already achieves strong results. The two-stage decomposition is noted here as the natural extension point.

### Feature Set

Core features (all derivable from data we have or can ingest):

| Feature | Source | Available? |
|---------|--------|------------|
| `age` | `player.birth_date` | Yes (93%) |
| `prior_pa` / `prior_ip` | `batting_stats` / `pitching_stats` | Yes |
| `prior_war` | `batting_stats.war` / `pitching_stats.war` | Yes |
| `prior_il_days` | IL transactions (new) | **No — Phase 1** |
| `starter_status` | `pitching_stats.gs/g` ratio; position for batters | Partial |
| `prior_2yr_pa` / `prior_2yr_ip` | `batting_stats` / `pitching_stats` (lag 2) | Yes |
| `il_stints_3yr` | IL transactions (new) | **No — Phase 1** |
| `career_il_days` | IL transactions (new) | **No — Phase 1** |
| `position` | `player.position` | Partial (needs enrichment) |

### Playing Time Aging Curve

A separate aging curve for playing time (distinct from the rate-stat aging curve Marcel already applies). Research shows:

- Hitters peak availability at ~27, with accelerating decline after 32.
- Pitchers peak availability at ~26, with sharper decline than hitters.
- The decline is nonlinear — roughly quadratic after peak.

The initial implementation uses a piecewise-linear curve fit to historical games-played data, parameterized identically to the existing `MarcelConfig` aging fields so the two curves can be tuned independently.

### Output

The model produces:

- **Point estimate:** projected PA (batters) or IP (pitchers).
- **Distribution:** p10/p25/p50/p75/p90 bounds on playing time, stored via the existing `projection_distribution` table. Playing time uncertainty is typically right-skewed (injury risk creates a long left tail), so the distribution will generally not be symmetric around the point estimate.

### Integration

The playing time model plugs into the existing system at two points:

1. **Marcel replacement.** `project_playing_time()` currently lives in `models/marcel/engine.py`. The new model can either replace this function or be a standalone model whose output Marcel (and other rate-stat models) consume as an input.
2. **Ensemble input.** The ensemble model can blend playing time projections from multiple sources (Marcel, the new model, third-party systems) just as it blends rate stats.

The standalone model approach is cleaner — it keeps Marcel purely about rate stats and lets the playing time model evolve independently. Marcel's `project_player()` would accept projected PT as a parameter rather than computing it internally.

---

## Phases

### Phase 1 — IL Transaction Ingest

Bring injury history into the database. This is the blocking prerequisite for everything else.

#### 1a. Domain type

**File:** `domain/il_stint.py` (new)

```python
@dataclass(frozen=True)
class ILStint:
    player_id: int
    season: int
    il_type: str          # "10-day", "15-day", "60-day"
    injury_location: str   # "elbow", "hamstring", "oblique", etc.
    start_date: str        # YYYY-MM-DD
    end_date: str | None   # None if still active
    days: int | None       # Computed or provided
    id: int | None = None
    loaded_at: str | None = None
```

#### 1b. Database migration

**File:** `db/migrations/NNN_il_stint.sql` (new)

```sql
CREATE TABLE il_stint (
    id              INTEGER PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES player(id),
    season          INTEGER NOT NULL,
    il_type         TEXT NOT NULL,
    injury_location TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    end_date        TEXT,
    days            INTEGER,
    loaded_at       TEXT,
    UNIQUE(player_id, start_date, injury_location)
);
```

#### 1c. Repository

**File:** `repos/il_stint_repo.py` (new)

Protocol `ILStintRepo`:
- `upsert(stint: ILStint, conn: Connection) -> ILStint`
- `get_by_player(player_id: int, conn: Connection) -> list[ILStint]`
- `get_by_player_season(player_id: int, season: int, conn: Connection) -> list[ILStint]`
- `get_season_totals(season: int, conn: Connection) -> list[tuple[int, int]]` — returns `(player_id, total_il_days)` pairs

Implement as `SqliteILStintRepo`.

#### 1d. Data source and ingest pipeline

**File:** `ingest/mlb_transactions_source.py` (new)

Fetch IL transactions from the MLB Stats API transactions endpoint:
```
GET https://statsapi.mlb.com/api/v1/transactions?startDate=YYYY-01-01&endDate=YYYY-12-31&transactionTypes=Injured List
```

Parse the JSON response into `ILStint` objects. Match players by `mlbam_id`. Compute `days` from `start_date` and `end_date` when not provided.

**File:** `ingest/column_maps.py`

Add `make_il_stint_mapper()` to map API response fields to `ILStint` fields.

**CLI:** Add `fbm ingest il [--season YEAR]` command that fetches and loads IL stints for one or more seasons.

#### 1e. Tests

- Round-trip: insert an `ILStint`, retrieve by player, verify fields match.
- `upsert` idempotency: re-inserting the same stint (same player + start_date + location) updates rather than duplicates.
- `get_season_totals` aggregates correctly across multiple stints per player.
- Column mapper correctly extracts injury location and IL type from API response format.
- Integration test: mock API response → loader → database → query.

---

### Phase 2 — Playing Time Features

Build the feature infrastructure to feed historical playing time, injury, age, and performance data into a model.

#### 2a. Feature source for IL data

**File:** `features/types.py`

Add `Source.IL_STINT` to the `Source` enum. This enables the feature DSL to reference IL data:

```python
il = SourceRef(Source.IL_STINT)
il.col("days").lag(1).alias("il_days_1")
il.col("days").lag(2).alias("il_days_2")
```

#### 2b. SQL generation for IL source

**File:** `features/sql.py`

Extend the SQL generator to handle `Source.IL_STINT`. Since IL data is per-stint (multiple rows per player-season), the join must aggregate:

```sql
LEFT JOIN (
    SELECT player_id, season, COALESCE(SUM(days), 0) AS days,
           COUNT(*) AS stint_count
    FROM il_stint
    GROUP BY player_id, season
) ils1 ON ils1.player_id = spine.player_id
     AND ils1.season = spine.season - 1
```

The feature DSL surfaces `days` (total IL days) and `stint_count` (number of IL stints) per season. Lag works as with other sources.

#### 2c. Playing time feature builder

**File:** `models/playing_time/features.py` (new)

```python
def build_playing_time_features(lags: int = 3) -> list[Feature | DerivedTransformFeature]:
    """Declare the feature set for the playing time model."""
```

Features:
- `age` — from `player.birth_date`
- `pa_1`, `pa_2`, `pa_3` — lagged PA from batting stats
- `ip_1`, `ip_2`, `ip_3` — lagged IP from pitching stats
- `war_1`, `war_2` — lagged WAR
- `il_days_1`, `il_days_2`, `il_days_3` — lagged total IL days per season
- `il_stints_1`, `il_stints_2` — lagged count of IL stints
- `g_1`, `g_2` — lagged games played
- `gs_1` — lagged games started (pitchers)
- `position` — from player table

Derived transforms:
- `il_days_3yr` — sum of IL days over last 3 seasons
- `il_recurrence` — binary flag: had IL stint in both of last 2 seasons
- `pt_trend` — ratio of year-1 PA to year-2 PA (increasing or decreasing playing time)

#### 2d. Tests

- Feature builder produces expected feature names and sources.
- SQL generation for `Source.IL_STINT` produces correct aggregation subquery.
- Lagged IL features align correctly with the target season.
- Derived transforms compute `il_days_3yr`, `il_recurrence`, and `pt_trend` correctly from known inputs.

---

### Phase 3 — Playing Time Regression Model

Train and register a regression model that predicts PA/IP from the features built in Phase 2.

#### 3a. Model skeleton

**File:** `models/playing_time/model.py` (new)

```python
@register("playing_time")
class PlayingTimeModel:
    name = "playing_time"
    supported_operations = frozenset({"train", "predict"})
    artifact_type = "pickle"  # or "json" for coefficients
```

Follows the same registration and lifecycle pattern as Marcel and Ensemble.

#### 3b. Training engine

**File:** `models/playing_time/engine.py` (new)

```python
def train_playing_time_model(
    dataset: DataFrame,
    config: PlayingTimeConfig,
) -> PlayingTimeCoefficients:
    """Fit a linear regression predicting next-season PA/IP."""
```

**Approach:** Ordinary least squares regression (no external ML library needed — `numpy.linalg.lstsq` or a small statsmodels dependency). The model is intentionally simple:

```
projected_PA = β₀ + β₁·age + β₂·pa_1 + β₃·war_1 + β₄·il_days_1 + β₅·starter + ...
```

Separate models for batters and pitchers (different coefficients, different target variables).

**Config:**

```python
@dataclass(frozen=True)
class PlayingTimeConfig:
    seasons: tuple[int, ...]       # Training seasons
    min_pa: int = 50               # Minimum PA to include in training set
    min_ip: float = 10.0           # Minimum IP to include in training set
    features: list[str] | None = None  # Feature subset (None = all)
```

**Output:** `PlayingTimeCoefficients` dataclass containing fitted coefficients, feature names, intercept, and training-set R².

#### 3c. Prediction engine

```python
def predict_playing_time(
    features: dict[str, float],
    coefficients: PlayingTimeCoefficients,
) -> float:
    """Apply fitted coefficients to a feature vector."""
```

Clamp output to `[0, max_pa]` where `max_pa` is a config parameter (default: 750 PA for batters, 250 IP for starters, 90 IP for relievers).

#### 3d. Wire into CLI

The model registers as `playing_time`, so the existing CLI commands work:
- `fbm prepare playing_time` — materialize features
- `fbm train playing_time` — fit coefficients
- `fbm predict playing_time` — generate projections
- `fbm evaluate playing_time` — compare to actuals

#### 3e. Tests

- Training with known data produces expected coefficients (use a small synthetic dataset where the answer is analytically known).
- Prediction applies coefficients correctly.
- Output is clamped to valid range.
- Batter and pitcher models are trained separately.
- Missing features (e.g., player with no IL history) default to 0 IL days, not `None`.
- End-to-end: prepare → train → predict → evaluate cycle completes without error.

---

### Phase 4 — Playing Time Aging Curve

Add a playing time–specific aging curve, separate from Marcel's rate-stat aging curve.

#### 4a. Empirical curve fitting

**File:** `models/playing_time/aging.py` (new)

```python
def fit_playing_time_aging_curve(
    dataset: DataFrame,
    player_type: str,   # "batter" or "pitcher"
) -> AgingCurve:
    """Fit a piecewise-linear aging curve to historical games-played data."""

@dataclass(frozen=True)
class AgingCurve:
    peak_age: float
    improvement_rate: float   # Per-year multiplier before peak
    decline_rate: float       # Per-year multiplier after peak
    player_type: str
```

Uses the delta method (paired seasons for the same player) to estimate age-specific changes in games played, correcting for survivor bias by weighting by inverse probability of appearing in the next season.

#### 4b. Integration into model

The aging curve becomes an additional feature or a post-prediction adjustment. Two options:

- **As a feature:** Include `age_pt_factor` (the aging curve multiplier for the player's age) as an input to the regression. The model learns how much weight to give it.
- **As a post-adjustment:** Multiply the regression output by the aging curve factor. Simpler but less flexible.

Start with the feature approach — it lets the model learn whether age effects are already captured by other features (e.g., `il_days` correlates with age).

#### 4c. Tests

- Curve produces multiplier > 1.0 for ages below peak, < 1.0 for ages above peak.
- Curve is monotonically decreasing after peak age.
- Batter and pitcher curves have different peak ages and decline rates.
- Known synthetic data produces expected curve parameters.

---

### Phase 5 — Distributional Playing Time Output

Produce uncertainty bounds on playing time projections, leveraging the `projection_distribution` infrastructure from the distributional-projections roadmap.

#### 5a. Residual-based distribution

**File:** `models/playing_time/engine.py`

After fitting the regression, compute residuals on the training set. Model the residual distribution (likely right-skewed due to injury risk creating a long left tail in PA):

```python
def compute_pt_distribution(
    predicted_pt: float,
    residual_distribution: ResidualDistribution,
    player_features: dict[str, float],
) -> StatDistribution:
    """Produce a percentile summary for projected playing time."""
```

The residual variance may depend on features — players with recent IL history have wider distributions. A simple approach: compute residual variance separately for buckets (e.g., 0 prior IL days vs. 1+ prior IL days, age < 30 vs. 30+) and select the appropriate bucket.

#### 5b. Wire into predict

The `PlayingTimeModel.predict()` method populates `PredictResult.distributions` with a `StatDistribution` for `pa` (batters) or `ip` (pitchers). These are persisted to `projection_distribution` by the standard persist path.

#### 5c. Tests

- Distribution percentiles are ordered: p10 < p25 < p50 < p75 < p90.
- Players with recent IL history have wider distributions (p90 - p10 is larger).
- Older players have wider distributions than younger players.
- Distribution is stored and retrievable via `ProjectionRepo.get_distributions`.

---

### Phase 6 — Marcel Integration and Ensemble

Connect the playing time model to the rest of the projection pipeline.

#### 6a. Decouple Marcel from playing time

**File:** `models/marcel/engine.py`

Modify `project_player()` to accept an optional `projected_pt: float | None` parameter. When provided, skip the internal `project_playing_time()` call and use the supplied value instead. When `None`, fall back to the existing Marcel formula (backward compatible).

#### 6b. Pipeline wiring

**File:** `models/marcel/model.py`

When the playing time model has been run for the target season, `MarcelModel.predict()` reads its output from the `projection` table and passes it to `project_player()`. This requires the playing time model to run before Marcel in the pipeline.

#### 6c. Ensemble playing time blending

**File:** `models/ensemble/engine.py`

The ensemble model already blends stats including PA/IP. With the playing time model as a registered system, the ensemble can include it as a component:

```python
components = {
    "playing_time": 0.5,   # Our model
    "steamer": 0.25,       # Steamer's implicit PT
    "zips": 0.25,          # ZiPS's implicit PT
}
```

No code changes needed — the ensemble engine already handles arbitrary systems.

#### 6d. Tests

- `project_player()` with `projected_pt` parameter uses the supplied value.
- `project_player()` without `projected_pt` falls back to Marcel formula.
- End-to-end: playing_time predict → marcel predict (consuming PT) → ensemble predict.
- Regression test: Marcel projections without the playing time model match prior output exactly.

---

## Phase Order

```
Phase 1 (IL ingest)
  ↓
Phase 2 (features)
  ↓
Phase 3 (regression model)
  ↓
Phase 4 (aging curve)    Phase 5 (distributional output)
  [both depend on Phase 3, independent of each other]
  ↓                       ↓
Phase 6 (Marcel integration + ensemble)
  [depends on Phase 3; benefits from 4 and 5 but doesn't require them]
```

Phase 6 can start as soon as Phase 3 is done. Phases 4 and 5 refine the model's accuracy and output richness but are not prerequisites for integration.

---

## Out of Scope

- **Team-level roster constraints.** FanGraphs enforces ~700 PA per position and ~1500 IP per pitching staff, redistributing playing time within teams. This requires modeling 30 team rosters simultaneously — a substantial constraint-optimization problem that belongs in a separate roster-projection system.
- **Spring training and minor-league competition.** Predicting whether a prospect or AAAA player will win a roster spot requires depth chart modeling and subjective evaluation beyond what historical stats can provide.
- **In-season updates.** Re-projecting playing time after a mid-season injury or trade. The model projects pre-season full-season totals. In-season adjustment is a different problem with different data requirements.
- **Injury type/severity modeling.** The Hawkes process approach (modeling self-exciting injury recurrence by body part) is promising but adds significant complexity. Phase 1 ingests injury location data so this extension is possible later, but the initial model uses total IL days as the feature, not injury-type-specific features.
- **Contract and service time effects.** Players in option years, arbitration-eligible players, and those on expiring contracts face different incentive structures that affect playing time. These are hard to quantify systematically.
