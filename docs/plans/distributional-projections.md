# Distributional Projections — Roadmap

## Goal

Support projection systems and models that produce **distributions of outcomes** rather than single point estimates. A player projected for 30 HR with a tight distribution (low variance) is a fundamentally different draft asset than one projected for 30 HR with a wide distribution (high variance). The system currently has no way to capture or use this information.

---

## Problem Statement

Every projection in the system today is a point estimate — a single float per stat. This loses information in several ways:

- **Third-party systems publish percentiles.** PECOTA provides 10th/25th/50th/75th/90th percentile projections. We ingest only the median and discard the rest.
- **Ensemble disagreement is signal.** When Marcel projects 25 HR and Steamer projects 35 HR, the spread tells us something about uncertainty. The current ensemble model (model-composition Phase 2) averages them to 30 HR and throws the disagreement away.
- **Bayesian or bootstrap models natively produce posteriors.** Future models that use MCMC sampling or bootstrapped training sets produce full distributions, not points.
- **Valuation benefits from risk adjustment.** The SGP valuation model (valuation-model roadmap) could discount high-variance players or reward safe floors, but only if it has access to the distribution.

The goal is to make distributions a **first-class concept** that flows through the system alongside point estimates, without disrupting existing point-estimate-only models.

---

## Design

### Representation: Percentile Summary + Optional Parametric Fit

Distributions are stored as a **percentile summary** with an **optional parametric fit**. This balances flexibility with queryability:

```python
@dataclass(frozen=True)
class StatDistribution:
    """Distribution of a single stat for one player/season/system."""
    stat: str
    p10: float
    p25: float
    p50: float        # median — may differ from point estimate
    p75: float
    p90: float
    mean: float | None = None
    std: float | None = None
    family: str | None = None   # "normal", "lognormal", "beta", etc.
```

Percentiles are always present (they can be derived from parametric fits or from empirical samples). Parametric fields are optional — useful for downstream consumers that want to do math with the distribution, but not required.

### Storage: Separate `projection_distribution` Table

A new table keyed on `(projection_id, stat)`:

```sql
CREATE TABLE projection_distribution (
    id            INTEGER PRIMARY KEY,
    projection_id INTEGER NOT NULL REFERENCES projection(id),
    stat          TEXT NOT NULL,
    p10           REAL NOT NULL,
    p25           REAL NOT NULL,
    p50           REAL NOT NULL,
    p75           REAL NOT NULL,
    p90           REAL NOT NULL,
    mean          REAL,
    std           REAL,
    family        TEXT,
    UNIQUE(projection_id, stat)
);
```

**Why a separate table rather than extending `projection`:**

- Point-estimate models (Marcel, third-party imports) don't produce distributions. A separate table means they need zero changes — no nullable columns, no schema migration on the hot path.
- One projection has many stat distributions (one row per stat), so it's a natural 1:N relationship.
- Queryable via standard SQL joins — no `json_extract` needed.
- The `projection` table's flat-column layout is already wide; adding 5+ columns per stat would be unwieldy.

### Domain Integration

The `Projection` domain type gains an optional field:

```python
@dataclass(frozen=True)
class Projection:
    # ... existing fields ...
    distributions: dict[str, StatDistribution] | None = None
```

When distributions are present, they ride alongside the point estimate. When absent (Marcel, most third-party imports), the field is `None` and nothing changes.

### Model Output

`PredictResult` gains an optional field:

```python
@dataclass
class PredictResult:
    model_name: str
    predictions: list[dict[str, Any]]
    output_path: str
    distributions: list[dict[str, Any]] | None = None
```

Each entry in `distributions` maps a `(player_id, stat)` pair to its distribution summary. Models that don't produce distributions leave it `None`.

### Feature System Extension

New `FeatureBuilder` methods for accessing distribution data:

```python
projection.col("hr").system("pecota").percentile(90).alias("pecota_hr_p90")
projection.col("hr").system("pecota").percentile(10).alias("pecota_hr_p10")
projection.col("hr").system("pecota").std().alias("pecota_hr_std")
```

These generate SQL that joins `projection_distribution` instead of `projection`:

```sql
LEFT JOIN projection_distribution pd0
    ON pd0.projection_id = pr0.id AND pd0.stat = 'hr'
-- select pd0.p90 AS pecota_hr_p90
```

### Source Patterns

Three ways distributions enter the system:

1. **Third-party ingest** — Systems like PECOTA publish percentile projections. The ingest pipeline parses them into `StatDistribution` objects alongside the point estimate.

2. **Ensemble spread** — The ensemble model (model-composition Phase 2) can compute distributional spread from component disagreement. Given N systems' projections for a stat, the ensemble can report p10/p25/p50/p75/p90 from the weighted component values rather than just the weighted mean.

3. **Native model output** — A Bayesian or bootstrap model that produces posterior samples. A helper function converts samples to the percentile summary:

```python
def samples_to_distribution(
    stat: str,
    samples: Sequence[float],
) -> StatDistribution:
    """Convert Monte Carlo samples to a percentile summary."""
```

---

## Phases

### Phase 1 — Distribution domain type and persistence

Establish the data model and storage layer. No model changes yet.

#### 1a. `StatDistribution` type

**File:** `domain/projection.py`

Add the `StatDistribution` frozen dataclass. Add the optional `distributions` field to `Projection`.

#### 1b. Database migration

**File:** `db/migrations/NNN_projection_distribution.sql` (new)

Create the `projection_distribution` table with the schema above. Foreign key to `projection(id)`.

#### 1c. Repository layer

**File:** `repos/projection_repo.py`

Extend `ProjectionRepo` protocol:

```python
def upsert_distributions(self, projection_id: int, distributions: list[StatDistribution]) -> None: ...
def get_distributions(self, projection_id: int) -> list[StatDistribution]: ...
```

Implement in `SqliteProjectionRepo`. `upsert_distributions` uses INSERT OR REPLACE. `get_distributions` returns all distributions for a projection, keyed by stat.

Optionally extend `get_by_player_season` and `get_by_season` with a `include_distributions: bool = False` parameter. When `True`, join `projection_distribution` and populate `Projection.distributions`. Default `False` so existing call sites are unaffected.

#### 1d. Tests

- Round-trip test: insert a projection with distributions, retrieve it, verify percentiles match.
- Test `upsert_distributions` is idempotent (re-inserting same stat updates the row).
- Test `get_distributions` returns empty list for a projection with no distributions.
- Test `get_by_player_season(include_distributions=True)` populates the `distributions` dict.
- Test `get_by_player_season(include_distributions=False)` leaves `distributions` as `None`.

---

### Phase 2 — Third-party distribution ingest

Enable importing percentile projections from systems that publish them.

#### 2a. Column map support for percentile columns

**File:** `ingest/column_maps.py`

Extend column maps to recognize percentile-suffixed stat columns. Convention: `{stat}_p{N}` (e.g., `hr_p10`, `hr_p90`). The column mapper groups these into `StatDistribution` objects per stat.

```python
def extract_distributions(
    row: dict[str, Any],
    percentile_columns: dict[str, list[str]],
) -> list[StatDistribution]:
    """Given a row with columns like hr_p10, hr_p50, hr_p90, build StatDistribution objects."""
```

#### 2b. Source adapter changes

**File:** `ingest/pybaseball_source.py` (or a new source for PECOTA-like systems)

When a source provides percentile columns, the adapter produces `Projection` objects with `distributions` populated. The persist step calls both `upsert()` (for the point estimate) and `upsert_distributions()` (for the distribution data).

#### 2c. Tests

- Test `extract_distributions` with a row containing percentile columns for multiple stats.
- Test that a row with no percentile columns produces an empty distribution list.
- Integration test: ingest a fake PECOTA-like CSV with percentile columns, verify both `projection` and `projection_distribution` tables are populated.

---

### Phase 3 — Ensemble distribution from component spread

Extend the ensemble model to produce distributional information from the disagreement among its component systems.

#### 3a. Ensemble distribution engine

**File:** `models/ensemble/engine.py`

Add a function that computes a distribution from weighted component projections:

```python
def weighted_spread(
    projections: list[tuple[dict[str, float], float]],
    stats: Sequence[str],
) -> dict[str, StatDistribution]:
    """Compute percentile summary from weighted component projections.

    For each stat, treat each system's projection as a point in the distribution,
    weighted by its ensemble weight. Compute weighted percentiles (p10–p90),
    weighted mean, and weighted standard deviation.
    """
```

When there are few components (e.g., 3 systems), the percentiles are rough — document this limitation. With more components or when components themselves provide distributions, the estimates improve.

#### 3b. Wire into ensemble predict

**File:** `models/ensemble/model.py`

After computing the weighted-average point estimate, also compute `weighted_spread` for each player. Attach the distributions to `PredictResult.distributions`. The persist step stores them in `projection_distribution`.

#### 3c. Tests

- Unit test `weighted_spread` with known weights and values.
- Test with 2 systems: spread reflects the disagreement range.
- Test with uniform weights: percentiles match simple quantiles of the component values.
- Integration test: ensemble predict produces both point estimates and distributions.

---

### Phase 4 — Feature system integration

Make distribution data accessible through the feature DSL and SQL generator.

#### 4a. `FeatureBuilder` extensions

**File:** `features/types.py`

Add methods to `FeatureBuilder`:

```python
def percentile(self, p: int) -> FeatureBuilder:
    """Access a percentile of the projection distribution (e.g., 10, 25, 50, 75, 90)."""

def std(self) -> FeatureBuilder:
    """Access the standard deviation of the projection distribution."""
```

These set new fields on the `Feature` dataclass (e.g., `distribution_percentile: int | None`, `distribution_stat: str | None`). Validation: only valid when `source == Source.PROJECTION` and `system` is set.

#### 4b. SQL generation

**File:** `features/sql.py`

When a feature has `distribution_percentile` set, the SQL generator:

1. Adds a join to `projection_distribution` (aliased `pd0`, `pd1`, etc.) on `projection_id` and `stat`.
2. Selects the appropriate column (`p10`, `p25`, etc.).

The join planner deduplicates: multiple features referencing different percentiles of the same stat/system share the same `projection_distribution` join.

#### 4c. Tests

- Unit test: SQL expression for `projection.col("hr").system("steamer").percentile(90)` produces correct join and column reference.
- Integration test: seed `projection` and `projection_distribution`, query via feature set, verify correct values are returned.
- Test that mixing point-estimate and distribution features in the same `FeatureSet` works (joins both tables).
- Test validation: `.percentile()` on a non-projection source raises `ValueError`.

---

### Phase 5 — Samples-to-distribution helper

Support models that produce Monte Carlo samples by providing a helper that converts samples to the standard percentile summary.

#### 5a. Conversion utility

**File:** `models/distributions.py` (new)

```python
def samples_to_distribution(
    stat: str,
    samples: Sequence[float],
) -> StatDistribution:
    """Convert a sequence of Monte Carlo samples to a StatDistribution.

    Computes p10/p25/p50/p75/p90 from empirical quantiles.
    Computes mean and std from the sample.
    Optionally fits a parametric family (normal, lognormal) and reports
    the best-fitting family name.
    """
```

Also provide a batch version:

```python
def samples_to_distributions(
    stat_samples: dict[str, Sequence[float]],
) -> dict[str, StatDistribution]:
    """Convert samples for multiple stats into StatDistribution objects."""
```

#### 5b. Tests

- Test with normally-distributed samples: verify percentiles approximate theoretical values.
- Test with small sample size (e.g., 10): verify no crash, percentiles are reasonable.
- Test with skewed data: verify percentiles reflect asymmetry.
- Test batch conversion produces one `StatDistribution` per stat.

---

## Phase Order

```
Phase 1 (domain + persistence)
  ↓
Phase 2 (third-party ingest)     Phase 3 (ensemble spread)     Phase 5 (samples helper)
  [all three depend on Phase 1, independent of each other]
  ↓                               ↓
Phase 4 (feature system integration)
  [depends on Phase 1; benefits from 2/3 for testing but doesn't require them]
```

Phases 2, 3, and 5 are independent and can be done in any order. Phase 4 only needs Phase 1 for the schema and types, but having real distribution data from Phase 2 or 3 makes integration testing more meaningful.

---

## Out of Scope

- **Full posterior storage** — Storing the complete set of Monte Carlo samples (N draws per player per stat) rather than just the summary. Storage-heavy and rarely needed; the percentile summary captures the useful information. If a model needs to persist raw samples, it can write to a model-specific artifact file.
- **Correlated joint distributions** — Capturing the correlation structure between stats (a player's high-HR world is also their high-K world). This requires storing covariance matrices or joint samples, which is substantially more complex. The percentile summary is per-stat and assumes independence. Joint simulation is a separate problem.
- **Distribution visualization** — Fan charts, violin plots, or other visual representations of projection distributions. Useful for a UI layer but not part of the core data model.
- **Confidence-weighted rankings** — Using distribution width to adjust draft rankings (e.g., preferring safe floors in early rounds, upside in late rounds). This is a valuation-layer concern that builds on this infrastructure but belongs in the valuation model roadmap.
- **Distribution comparison metrics** — KL divergence, Wasserstein distance, or CRPS for evaluating distributional predictions against actuals. Valuable for evaluation but a separate effort once distributions are flowing.
