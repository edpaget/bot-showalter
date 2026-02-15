# Distributional Projections

The system supports projection distributions alongside point estimates. A player projected for 30 HR with a tight distribution is a different draft asset than one projected for 30 HR with a wide distribution. Distributions flow through the entire stack: ingestion, persistence, ensemble blending, feature engineering, and model output.

## Representation

Distributions use a percentile summary with optional summary statistics:

```python
@dataclass(frozen=True)
class StatDistribution:
    stat: str
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    mean: float | None = None
    std: float | None = None
    family: str | None = None   # reserved for future parametric fitting
```

Percentiles are always present. Mean, std, and family are optional. The `Projection` domain type carries distributions as an optional field:

```python
@dataclass(frozen=True)
class Projection:
    # ... existing fields ...
    distributions: dict[str, StatDistribution] | None = None
```

When distributions are absent (e.g., Marcel projections), the field is `None` and nothing changes for point-estimate-only code paths.

## Storage

Distributions live in a dedicated `projection_distribution` table with a foreign key to `projection`:

```sql
CREATE TABLE projection_distribution (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
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

A separate table keeps the `projection` schema unchanged for point-estimate-only models. One projection has many stat distributions (one row per stat), a natural 1:N relationship queryable with standard SQL joins.

### Repository API

`ProjectionRepo` provides distribution methods:

```python
# Store distributions for a projection
repo.upsert_distributions(projection_id, distributions)

# Retrieve distributions for a projection
distributions = repo.get_distributions(projection_id)

# Load projections with distributions attached
projections = repo.get_by_player_season(player_id, season, include_distributions=True)
```

The `include_distributions` flag defaults to `False` so existing call sites pay no cost. When `True`, the repo batch-fetches from `projection_distribution` and populates each `Projection.distributions` dict.

## Three source patterns

Distributions enter the system through three paths.

### 1. Third-party ingest

Systems like PECOTA publish percentile projections. The ingest pipeline recognizes percentile-suffixed columns in the source data.

```python
from fantasy_baseball_manager.ingest.column_maps import extract_distributions
```

`extract_distributions(row, column_map)` scans for columns named `{csv_col}_p10`, `{csv_col}_p25`, etc. When all five required percentiles are present for a stat, it builds a `StatDistribution`. Optional `_mean` and `_std` suffixed columns are included when available. Stats with incomplete percentile data are skipped.

The projection row mappers (`make_fg_projection_batting_mapper`, `make_fg_projection_pitching_mapper`) call `extract_distributions` automatically and attach the results to the `Projection.distributions` field.

### 2. Ensemble spread

The ensemble model computes distributional spread from disagreement among component projection systems.

```python
from fantasy_baseball_manager.models.ensemble.engine import weighted_spread

spread = weighted_spread(projections, stats=["hr", "rbi", "avg"])
# spread["hr"] is a StatDistribution
```

`weighted_spread` takes a sequence of `(stat_json, weight)` pairs — one per component system — and for each stat:

1. Sorts values and builds a weighted cumulative distribution using midpoint convention.
2. Extracts p10/p25/p50/p75/p90 via linear interpolation on the CDF.
3. Computes weighted mean and population standard deviation.

Returns an empty dict for stats with fewer than 2 contributing systems (no spread to measure). The ensemble model calls this during prediction and stores the result in `projection_distribution`.

### 3. Monte Carlo samples

Models that produce posterior samples (Bayesian inference, bootstrap ensembles) convert them to distributions via a helper:

```python
from fantasy_baseball_manager.models.distributions import (
    samples_to_distribution,
    samples_to_distributions,
)

dist = samples_to_distribution("hr", samples)
# dist.p10, dist.p50, dist.p90, dist.mean, dist.std

batch = samples_to_distributions({"hr": hr_samples, "rbi": rbi_samples})
```

`samples_to_distribution` computes empirical percentiles via `numpy.percentile` (default linear interpolation) and sample mean/std (`ddof=0`). Requires at least 2 samples. The `family` field is left `None` — no parametric fitting is performed.

## Feature system integration

Distribution data is accessible through the feature DSL, enabling models to use percentiles and uncertainty measures as training features.

### Declaring distribution features

```python
from fantasy_baseball_manager.features import projection

feature_set = FeatureSet(
    name="uncertainty_features",
    features=(
        projection.col("hr").system("pecota").percentile(90).alias("pecota_hr_p90"),
        projection.col("hr").system("pecota").percentile(10).alias("pecota_hr_p10"),
        projection.col("hr").system("pecota").std().alias("pecota_hr_std"),
        # Mix point estimates and distributions in the same feature set
        projection.col("hr").system("steamer").alias("steamer_hr"),
    ),
    seasons=(2024,),
)
```

`FeatureBuilder.percentile(p)` accepts `p` in `{10, 25, 50, 75, 90}`. Both `.percentile()` and `.std()` raise `ValueError` on non-projection sources.

### SQL generation

When a feature references a distribution column, the SQL generator:

1. Plans a `LEFT JOIN` to `projection_distribution` (aliased `pd0`, `pd1`, etc.) on `projection_id` and `stat`.
2. Selects the appropriate column (`p10`, `p25`, `p50`, `p75`, `p90`, `mean`, or `std`).
3. Deduplicates: multiple features referencing different columns of the same stat/system share one join.

```sql
LEFT JOIN projection_distribution pd0
    ON pd0.projection_id = pr0.id AND pd0.stat = 'hr'
-- SELECT pd0.p90 AS pecota_hr_p90, pd0.p10 AS pecota_hr_p10
```

The dataset assembler handles this transparently — distribution features are standard columns in the materialized dataset table.

## Key design decisions

- **Percentile summary, not raw samples.** Storing N draws per player per stat would be storage-heavy and rarely needed. The five-percentile summary captures the useful information.
- **Per-stat independence.** Distributions are stored per stat. Correlation structure between stats (a player's high-HR scenario is also their high-K scenario) is out of scope — joint simulation is a separate problem.
- **Separate table.** Point-estimate-only models need zero changes. No nullable columns on the main `projection` table, no schema migration on the hot path.
- **Lazy loading.** Distributions are only fetched when `include_distributions=True` is passed, keeping the default query path fast.
- **No parametric fitting.** The `family` field exists for future use, but no phase currently fits parametric families (normal, lognormal, etc.). All distributions are empirical.
