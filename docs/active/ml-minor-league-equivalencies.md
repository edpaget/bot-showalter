# ML-Based Minor League Equivalencies

## Overview

Replace hand-tuned translation factors with a machine learning model that learns MiLB → MLB stat translations from historical call-up data. The model learns level-specific, stat-specific, and context-aware translations that capture the nuances traditional MLE constants miss.

---

## Problem Statement

The current projection system has a blind spot for players with limited MLB history:

| MLB PA | Current Behavior | Problem |
|--------|------------------|---------|
| 0 | No projection | Miss on rookies entirely |
| 1-50 | Heavy regression to mean | Ignores MiLB signal |
| 50-200 | Partial regression | Still underweights MiLB track record |
| 200+ | Normal Marcel | Works well |

Traditional MLE uses fixed multipliers (AAA → 0.90, AA → 0.80, etc.) that:
- Assume all stats translate equally (they don't—K% translates better than HR/FB)
- Ignore player age (a 22-year-old at AA is different from a 26-year-old)
- Ignore league/park effects within levels (PCL vs IL at AAA)
- Ignore modern minor league changes (2021 ball deadening, universal DH, pitch clock)
- Can't leverage MiLB Statcast data (available since 2022 at AAA)

An ML model can learn these nuances from data.

---

## Data Sources

### Primary: Minor League Statistics

**Source**: MLB Stats API (direct HTTP requests)

pybaseball does **not** have MiLB stats functions. Instead, we query the MLB Stats API directly using `sportId` to specify the level:

| Level | Sport ID | Code |
|-------|----------|------|
| Triple-A | 11 | `aaa` |
| Double-A | 12 | `aax` |
| High-A | 13 | `afa` |
| Single-A | 14 | `afx` |
| Rookie | 16 | `rok` |

```python
# Fetch all AAA batting stats for 2024
import requests

response = requests.get(
    "https://statsapi.mlb.com/api/v1/stats",
    params={
        "stats": "season",
        "season": 2024,
        "sportId": 11,  # AAA
        "group": "hitting",
        "limit": 1000,
    }
)
data = response.json()
players = data["stats"][0]["splits"]  # List of player-season records
```

**Available fields**: PA, AB, H, 2B, 3B, HR, BB, SO, SB, CS, HBP, SF, AVG, OBP, SLG, BABIP, age, plus team/league metadata

**Coverage**: 2021-present (2020 had no MiLB season due to COVID). Sample sizes per year at AAA: ~100-140 qualifying players.

**Player IDs**: Returns MLBAM IDs that match MLB data, enabling direct crosswalk.

### Secondary: MLB Debut Outcomes

**Source**: pybaseball `batting_stats()` / `pitching_stats()` (already integrated)

**Training target**: Actual MLB rates in year of call-up and year after

### Tertiary: MiLB Statcast (AAA only, 2023+)

**Source**: Baseball Savant Minor League Search (`baseballsavant.mlb.com/statcast-search-minors`)

All AAA ballparks have had Statcast since 2023. Data available via:
- Baseball Savant CSV exports
- Pitch-level scraping (same pattern as MLB Statcast)

**Available**: xBA, xSLG, xwOBA, barrel%, hard-hit%, sprint speed, launch angle, exit velocity

**Value**: High-signal features for recent call-ups; mirrors MLB Statcast integration. Optional enhancement—model can work without this initially.

### Supplementary: Context Data

| Data | Source | Purpose |
|------|--------|---------|
| Player age | MLB Stats API (included in response) | Age-adjusted translations |
| League | MLB Stats API (PCL vs IL at AAA) | League-specific adjustments |
| Org/affiliate | MLB Stats API team → parent mapping | Org development effects |

---

## Training Data Construction

### Population: Historical Call-Ups (2021-2024)

**Inclusion criteria**:
- Player had ≥200 PA at AAA or AA in year Y-1 or Y
- Player debuted or had <200 MLB PA entering year Y
- Player accumulated ≥100 MLB PA in year Y or Y+1

**Exclusion criteria**:
- Players with prior MLB experience (>200 PA before MiLB season)
- September call-ups with <50 PA (too noisy)

**Data availability note**: MiLB data via MLB Stats API is available from 2021+ (2020 had no MiLB season). This limits our training window but provides clean, consistent data.

**Expected sample size**: ~100-150 qualifying batter-seasons per year × 4 years = ~400-600 samples. Smaller than ideal, but gradient boosting handles this well with proper regularization.

### Handling Multi-Level Seasons

When a player plays at multiple levels in a single season (e.g., 300 PA at AA + 200 PA at AAA), we create **PA-weighted aggregate features**:

```python
def aggregate_multi_level_stats(seasons: list[MiLBSeasonStats]) -> dict:
    """Aggregate stats across levels, weighted by PA."""
    total_pa = sum(s.pa for s in seasons)

    # PA-weighted rates
    weighted_hr_rate = sum(s.hr / s.pa * s.pa for s in seasons) / total_pa
    weighted_so_rate = sum(s.so / s.pa * s.pa for s in seasons) / total_pa
    # ... etc for other rate stats

    # Highest level reached (categorical feature)
    highest_level = min(s.sport_id for s in seasons)  # Lower ID = higher level

    # Level distribution (how much time at each level)
    pct_at_aaa = sum(s.pa for s in seasons if s.sport_id == 11) / total_pa
    pct_at_aa = sum(s.pa for s in seasons if s.sport_id == 12) / total_pa

    return {
        "hr_rate": weighted_hr_rate,
        "so_rate": weighted_so_rate,
        "total_pa": total_pa,
        "highest_level": highest_level,
        "pct_at_aaa": pct_at_aaa,
        "pct_at_aa": pct_at_aa,
        # ...
    }
```

This approach:
1. Preserves signal from all levels (more data = better)
2. Lets the model learn that level distribution matters (promoted mid-season vs stuck at AA)
3. Avoids arbitrary choices about which level to use

### Feature/Target Alignment

```
Features (from year Y-1 MiLB season):
├── MiLB rate stats (HR/PA, SO/PA, BB/PA, etc.)
├── Level indicators (AAA, AA, A+, A)
├── Age at end of MiLB season
├── MiLB Statcast (if available)
├── Context (org, park, prospect rank)
└── Sample size (MiLB PA)

Target (from year Y or Y+1 MLB):
├── MLB rate stats (HR/PA, SO/PA, BB/PA, etc.)
└── Weighted by MLB PA (more PA = more reliable target)
```

### Train/Validation/Test Split

- **Train**: 2021-2022 call-ups (~200-300 samples)
- **Validation**: 2023 call-ups (~100-150 samples) for hyperparameter tuning
- **Test**: 2024 call-ups (~100-150 samples) for final evaluation

Temporal split prevents data leakage and tests generalization. The smaller dataset requires careful regularization to avoid overfitting.

---

## Feature Engineering

### Batter Features (MiLB)

**Rate stats** (per PA):
- `hr_rate`: HR / PA
- `so_rate`: SO / PA
- `bb_rate`: BB / PA
- `hit_rate`: H / PA (proxy for contact)
- `xbh_rate`: (2B + 3B) / PA
- `sb_rate`: SB / (H + BB + HBP)
- `iso`: SLG - AVG

**Level encoding**:
- `level_aaa`: 1 if highest level is AAA, else 0
- `level_aa`: 1 if highest level is AA, else 0
- `level_a_plus`: 1 if highest level is A+, else 0
- `level_a`: 1 if highest level is A or below, else 0

**Age features**:
- `age`: Age at end of season
- `age_squared`: Captures non-linear age effects
- `age_for_level`: Age relative to typical age at level (e.g., 22 at AA is young, 26 is old)

**Sample size**:
- `milb_pa`: Total MiLB PA (model can learn to trust larger samples more)
- `log_milb_pa`: Log-transformed PA

**Statcast features** (AAA 2023+ only, else imputed or omitted):
- `xba`, `xslg`, `xwoba`
- `barrel_rate`, `hard_hit_rate`
- `sprint_speed`
- `has_statcast`: Indicator for whether Statcast data available

**Context features** (derived from stats, no scouting data):
- `league_id`: PCL vs IL at AAA (encodes park/league environment)
- `pct_at_aaa`: Fraction of PA at AAA (vs lower levels)
- `pct_at_aa`: Fraction of PA at AA

### Pitcher Features (MiLB)

**Rate stats** (per batter faced or per IP):
- `so_rate`: SO / BF (or SO/9)
- `bb_rate`: BB / BF
- `hr_rate`: HR / BF
- `hit_rate`: H / BF
- `era`: Traditional ERA (for context)
- `whip`: WHIP

**Role indicator**:
- `is_starter`: 1 if GS > G/2, else 0 (relievers may translate differently)

**Same level, age, sample size, Statcast, and context features as batters.**

### Feature Imputation

For missing Statcast data (pre-2022 or non-AAA):
- Impute from rate stats using a learned mapping: `xba ≈ f(hit_rate, so_rate, level)`
- Or use indicator + zero-imputation (let model learn to ignore)

---

## Model Architecture

### Option A: Gradient Boosting (Recommended Starting Point)

**Rationale**: Proven on residual prediction task, handles mixed feature types well, interpretable via SHAP.

```python
class MLEGradientBoostingModel:
    """Per-stat LightGBM models for MiLB → MLB translation."""

    def __init__(self, target_stats: list[str]):
        self.models: dict[str, lgb.Booster] = {}
        self.target_stats = target_stats  # ['hr_rate', 'so_rate', 'bb_rate', ...]

    def fit(self, X: np.ndarray, y: dict[str, np.ndarray], sample_weight: np.ndarray):
        for stat in self.target_stats:
            self.models[stat] = lgb.train(
                params={
                    'objective': 'regression',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'n_estimators': 200,
                    'min_child_samples': 20,  # Regularization for small dataset
                },
                train_set=lgb.Dataset(X, y[stat], weight=sample_weight),
                # ... validation, early stopping
            )

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {stat: model.predict(X) for stat, model in self.models.items()}
```

**Hyperparameters to tune**:
- `num_leaves`: 15-63 (controls complexity)
- `min_child_samples`: 10-50 (regularization)
- `learning_rate`: 0.01-0.1
- `n_estimators`: 100-500 (with early stopping)

### Option B: Multi-Task Neural Network

**Rationale**: Exploit correlations between stats (K% and BB% translate together). Follows existing MTL pattern.

```python
class MLEMultiTaskNet(nn.Module):
    """Shared trunk with stat-specific heads for MLE prediction."""

    def __init__(self, input_dim: int, target_stats: list[str]):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            stat: nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            for stat in target_stats
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.trunk(x)
        return {stat: head(shared).squeeze(-1) for stat, head in self.heads.items()}
```

**Loss function**: Weighted MSE with sample-size weighting:
```python
loss = sum(
    (mlb_pa / mean_pa) * (pred[stat] - actual[stat])**2
    for stat in target_stats
)
```

### Option C: Ensemble

Combine GB and MTL predictions (analogous to `marcel_mtl` blending):
```python
final = 0.6 * gb_prediction + 0.4 * mtl_prediction
```

### Recommendation

**Start with Option A (Gradient Boosting)**:
1. Simpler to implement and debug
2. Matches existing `residual_model.py` patterns
3. SHAP feature importance reveals what drives translations
4. Can add MTL ensemble later if GB shows promise

---

## Integration Architecture

### New Module Structure

```
src/fantasy_baseball_manager/minors/
├── __init__.py
├── types.py              # MinorLeagueSeasonStats, MLEPrediction dataclasses
├── data_source.py        # MinorLeagueDataSource protocol + MLB Stats API impl
├── cached_data_source.py # CachedMinorLeagueDataSource wrapper
├── features.py           # MLEFeatureExtractor (mirrors ml/features.py pattern)
├── model.py              # MLEModel (GB or MTL)
├── training.py           # MLEModelTrainer (mirrors ml/training.py pattern)
├── persistence.py        # Model save/load (mirrors ml/mtl/persistence.py)
└── rate_computer.py      # MLERateComputer (RateComputer protocol impl)
```

### Data Types

```python
@dataclass
class MinorLeagueSeasonStats:
    player_id: str
    season: int
    level: str  # 'AAA', 'AA', 'A+', 'A', 'Rk'
    pa: int
    ab: int
    h: int
    doubles: int
    triples: int
    hr: int
    bb: int
    so: int
    sb: int
    cs: int
    # Optional Statcast (AAA 2022+)
    xba: float | None = None
    xslg: float | None = None
    barrel_rate: float | None = None
    hard_hit_rate: float | None = None
    sprint_speed: float | None = None


@dataclass
class MLEPrediction:
    player_id: str
    source_season: int
    source_level: str
    predicted_rates: dict[str, float]  # {'hr_rate': 0.035, 'so_rate': 0.22, ...}
    confidence: float  # Based on sample size and model uncertainty
```

### Caching

MiLB data is fetched from the MLB Stats API, which can be slow and rate-limited. We cache aggressively following the existing `CacheStore` pattern.

**Cache key structure**: `milb_batting:{year}:{sport_id}` (e.g., `milb_batting:2024:11` for AAA 2024)

**TTL strategy**:
- Historical seasons (year < current): 365 days (data won't change)
- Current season: 1 day (updates during season)

```python
# minors/data_source.py

class MinorLeagueDataSource(Protocol):
    """Protocol for fetching minor league statistics."""

    def batting_stats(self, year: int, sport_id: int) -> list[MinorLeagueSeasonStats]:
        """Fetch batting stats for a level/year."""
        ...

    def batting_stats_all_levels(self, year: int) -> list[MinorLeagueSeasonStats]:
        """Fetch batting stats across all MiLB levels for a year."""
        ...


class MLBStatsAPIDataSource:
    """Fetches MiLB stats from MLB Stats API."""

    SPORT_IDS = {"AAA": 11, "AA": 12, "A+": 13, "A": 14, "Rookie": 16}

    def batting_stats(self, year: int, sport_id: int) -> list[MinorLeagueSeasonStats]:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/stats",
            params={
                "stats": "season",
                "season": year,
                "sportId": sport_id,
                "group": "hitting",
                "limit": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        return self._parse_batting_response(response.json(), year, sport_id)

    def batting_stats_all_levels(self, year: int) -> list[MinorLeagueSeasonStats]:
        all_stats = []
        for level, sport_id in self.SPORT_IDS.items():
            all_stats.extend(self.batting_stats(year, sport_id))
        return all_stats
```

```python
# minors/cached_data_source.py

_NAMESPACE = "milb_stats"
_TTL_HISTORICAL = 60 * 60 * 24 * 365  # 1 year
_TTL_CURRENT = 60 * 60 * 24  # 1 day


class CachedMinorLeagueDataSource:
    """Wraps MinorLeagueDataSource with CacheStore-backed persistence."""

    def __init__(
        self,
        delegate: MinorLeagueDataSource,
        cache: CacheStore,
        current_year: int | None = None,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._current_year = current_year or datetime.now().year

    def batting_stats(self, year: int, sport_id: int) -> list[MinorLeagueSeasonStats]:
        key = f"batting:{year}:{sport_id}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            return _deserialize_milb_batting(cached)

        result = self._delegate.batting_stats(year, sport_id)
        ttl = _TTL_CURRENT if year >= self._current_year else _TTL_HISTORICAL
        self._cache.put(_NAMESPACE, key, _serialize_milb_batting(result), ttl)
        return result

    def batting_stats_all_levels(self, year: int) -> list[MinorLeagueSeasonStats]:
        key = f"batting_all:{year}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            return _deserialize_milb_batting(cached)

        result = self._delegate.batting_stats_all_levels(year)
        ttl = _TTL_CURRENT if year >= self._current_year else _TTL_HISTORICAL
        self._cache.put(_NAMESPACE, key, _serialize_milb_batting(result), ttl)
        return result


def _serialize_milb_batting(stats: list[MinorLeagueSeasonStats]) -> str:
    return json.dumps([asdict(s) for s in stats])


def _deserialize_milb_batting(data: str) -> list[MinorLeagueSeasonStats]:
    return [MinorLeagueSeasonStats(**row) for row in json.loads(data)]
```

**Training data caching**: The `MLETrainingDataCollector` should use cached data sources. Since training data spans multiple years, the first run will be slow, but subsequent runs (for hyperparameter tuning, retraining) will be instant.

### MLERateComputer

```python
class MLERateComputer:
    """RateComputer that uses ML model for players with limited MLB history."""

    def __init__(
        self,
        delegate: RateComputer,
        mle_model: MLEModel,
        minor_league_source: MinorLeagueDataSource,
        mlb_pa_threshold: int = 200,
    ):
        self.delegate = delegate
        self.mle_model = mle_model
        self.minor_league_source = minor_league_source
        self.mlb_pa_threshold = mlb_pa_threshold

    def compute_batting_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int = 3,
    ) -> list[PlayerRates]:
        # 1. Get MLB rates from delegate
        mlb_rates = self.delegate.compute_batting_rates(data_source, year, years_back)

        # 2. Identify players needing MLE
        low_pa_players = [p for p in mlb_rates if p.opportunities < self.mlb_pa_threshold]

        # 3. Fetch MiLB stats and predict
        for player in low_pa_players:
            milb_stats = self.minor_league_source.get_stats(player.player_id, year - 1)
            if milb_stats and milb_stats.pa >= 100:
                mle_pred = self.mle_model.predict(milb_stats)
                player.rates = self._blend_rates(
                    mlb_rates=player.rates,
                    mlb_pa=player.opportunities,
                    mle_rates=mle_pred.predicted_rates,
                    mle_pa=milb_stats.pa,
                )
                player.metadata['mle_applied'] = True
                player.metadata['mle_source_level'] = milb_stats.level

        return mlb_rates

    def _blend_rates(
        self,
        mlb_rates: dict[str, float],
        mlb_pa: int,
        mle_rates: dict[str, float],
        mle_pa: int,
    ) -> dict[str, float]:
        """Sample-size weighted blending."""
        total_pa = mlb_pa + mle_pa
        blended = {}
        for stat in mlb_rates:
            mlb_val = mlb_rates.get(stat, 0)
            mle_val = mle_rates.get(stat, mlb_val)  # Fall back to MLB if MLE missing
            blended[stat] = (mlb_pa * mlb_val + mle_pa * mle_val) / total_pa
        return blended
```

### Pipeline Integration

```python
# presets.py addition
def marcel_mle_pipeline() -> ProjectionPipeline:
    """Marcel pipeline with ML-based minor league equivalencies."""
    mle_model = load_mle_model()  # From persistence
    return (
        PipelineBuilder("marcel_mle", config=config)
        .with_mle_rate_computer(
            mle_model=mle_model,
            mlb_pa_threshold=200,
        )
        .with_park_factors()
        .with_component_aging()
        .with_statcast_adjustments()
        .with_gb_residuals()
        .build()
    )
```

---

## Training Pipeline

### Data Collection

```python
class MLETrainingDataCollector:
    """Collect (MiLB features, MLB outcomes) pairs for MLE model training."""

    def collect(
        self,
        milb_source: MinorLeagueDataSource,
        mlb_source: StatsDataSource,
        years: list[int],
        min_milb_pa: int = 200,
        min_mlb_pa: int = 100,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Dict of target arrays per stat
            sample_weights: MLB PA for weighting
        """
        samples = []

        for year in years:
            # Get all players with qualifying MiLB seasons
            milb_players = milb_source.get_qualifying_players(year, min_milb_pa)

            for player_id in milb_players:
                milb_stats = milb_source.get_stats(player_id, year)
                mlb_stats = mlb_source.get_debut_stats(player_id, year, year + 1)

                if mlb_stats and mlb_stats.pa >= min_mlb_pa:
                    samples.append({
                        'features': self._extract_features(milb_stats),
                        'targets': self._extract_targets(mlb_stats),
                        'weight': mlb_stats.pa,
                    })

        return self._to_arrays(samples)
```

### Training Orchestration

```python
class MLEModelTrainer:
    """Train and validate MLE model."""

    def train(
        self,
        train_years: list[int],
        val_years: list[int],
        model_type: Literal['gb', 'mtl'] = 'gb',
    ) -> MLEModel:
        # 1. Collect data
        X_train, y_train, w_train = self.collector.collect(train_years)
        X_val, y_val, w_val = self.collector.collect(val_years)

        # 2. Train model
        if model_type == 'gb':
            model = MLEGradientBoostingModel(TARGET_STATS)
            model.fit(X_train, y_train, w_train, X_val, y_val, w_val)
        else:
            model = MLEMultiTaskNet(X_train.shape[1], TARGET_STATS)
            # ... PyTorch training loop

        # 3. Evaluate on validation
        val_metrics = self._evaluate(model, X_val, y_val, w_val)
        print(f"Validation RMSE: {val_metrics}")

        return model
```

---

## Evaluation Methodology

### Metrics

| Metric | What it Measures |
|--------|------------------|
| **RMSE** | Raw prediction accuracy per stat |
| **Weighted RMSE** | RMSE weighted by MLB PA (prioritize reliable targets) |
| **Spearman ρ** | Rank correlation (did we identify the best prospects?) |
| **Calibration** | Are 80th percentile predictions actually 80th percentile? |

### Baselines

1. **No MLE**: Current system (heavy regression for low-PA players)
2. **Traditional MLE**: Fixed factors (AAA=0.90, AA=0.80, etc.)
3. **Marcel-only**: Marcel projection ignoring MiLB data

### Evaluation Protocol

```python
def evaluate_mle_model(
    model: MLEModel,
    test_years: list[int],
) -> dict[str, float]:
    """Evaluate on held-out call-up cohorts."""
    results = {}

    for year in test_years:
        # Get test set: players who debuted in this year
        X_test, y_test, w_test = collector.collect([year])

        # Predict
        y_pred = model.predict(X_test)

        # Compute metrics
        for stat in TARGET_STATS:
            rmse = weighted_rmse(y_pred[stat], y_test[stat], w_test)
            rho = spearmanr(y_pred[stat], y_test[stat]).correlation
            results[f'{stat}_rmse_{year}'] = rmse
            results[f'{stat}_rho_{year}'] = rho

    return results
```

### Success Criteria

The ML MLE model should:
1. **Beat traditional MLE** by ≥5% RMSE on rate stats
2. **Beat no-MLE baseline** by ≥10% RMSE for players with <100 MLB PA
3. **Maintain rank correlation** (Spearman ρ ≥ 0.5) for prospect ordering
4. **Not degrade** existing projections for players with sufficient MLB history

---

## Implementation Plan

### Phase 1: Data Infrastructure ✅ COMPLETE

1. ✅ Define `MinorLeagueBatterSeasonStats` and `MinorLeaguePitcherSeasonStats` dataclasses in `minors/types.py`
2. ✅ Implement `MinorLeagueDataSource` protocol and `MLBStatsAPIDataSource`
3. ✅ Implement `CachedMinorLeagueDataSource` wrapper with `CacheStore` integration
4. ✅ Build `MLETrainingDataCollector` with proper temporal alignment and multi-level aggregation
5. ✅ Write comprehensive tests (51 tests) for data collection, caching, and edge cases

**Deliverables**:
- ✅ `minors/types.py` - Includes `MinorLeagueLevel` enum, `MinorLeagueBatterSeasonStats`, `MinorLeaguePitcherSeasonStats`, `MLEPrediction`
- ✅ `minors/data_source.py` - Protocol + MLB Stats API implementation with response parsing
- ✅ `minors/cached_data_source.py` - Cache wrapper with TTL strategy (1 year historical, 1 day current)
- ✅ `minors/training_data.py` - `AggregatedMiLBStats` for multi-level seasons, `MLETrainingDataCollector` (refactored to use feature extractor in Phase 2)
- ✅ `tests/minors/` - 51 passing tests covering types, data sources, caching, and training data collection

**Implementation Notes**:
- `MinorLeagueLevel` enum maps sport IDs (AAA=11, AA=12, HIGH_A=13, SINGLE_A=14, ROOKIE=16)
- Training data uses PA-weighted aggregation for multi-level seasons
- Initial features: rate stats (12), age (2), level one-hot (4), level distribution (4), sample size (2) — expanded to 32 in Phase 2
- Temporal alignment: MiLB year Y-1 features → MLB year Y/Y+1 targets
- Qualifying criteria: ≥200 MiLB PA, ≥100 MLB PA, ≤200 prior MLB PA, AAA or AA level

### Phase 2: Feature Engineering ✅ COMPLETE

1. ✅ Implement `MLEBatterFeatureExtractor` with all batter features
2. ✅ Add Statcast feature handling (indicator + zero-imputation for missing)
3. ✅ Refactor `MLETrainingDataCollector` to use feature extractor

**Deliverables**:
- ✅ `minors/features.py` - `MLEBatterFeatureExtractor` with 32 features
- ✅ `minors/types.py` - Added `MiLBStatcastStats` dataclass for Statcast data
- ✅ `tests/minors/test_features.py` - 23 tests for feature extraction
- ✅ Updated `minors/training_data.py` to delegate to feature extractor

**Implementation Notes**:
- Feature extractor follows frozen dataclass pattern from `ml/features.py`
- 32 total features:
  - Rate stats (12): hr_rate, so_rate, bb_rate, hit_rate, singles_rate, doubles_rate, triples_rate, sb_rate, iso, avg, obp, slg
  - Age features (3): age, age_squared, age_for_level (relative to typical age at level)
  - Level one-hot (4): level_aaa, level_aa, level_high_a, level_single_a
  - Level distribution (4): pct_at_aaa, pct_at_aa, pct_at_high_a, pct_at_single_a
  - Sample size (2): total_pa, log_pa
  - Statcast (7): xba, xslg, xwoba, barrel_rate, hard_hit_rate, sprint_speed, has_statcast
- Statcast data handling: indicator + zero-imputation (has_statcast=0 when unavailable)
- `TYPICAL_AGE_BY_LEVEL` constant: AAA=25, AA=23, HIGH_A=22, SINGLE_A=21, ROOKIE=19
- `extract_batch()` method for efficient batch extraction with optional Statcast lookup

### Phase 3: Model Training ✅ COMPLETE

1. ✅ Implement `MLEGradientBoostingModel` with per-stat LightGBM models
2. ✅ Implement `MLEModelTrainer` for training orchestration
3. ✅ Implement `MLEModelStore` for model persistence
4. Hyperparameter tuning via cross-validation (deferred to actual training)
5. SHAP analysis for feature importance (available via `model.feature_importances()`)

**Deliverables**:
- ✅ `minors/model.py` - `MLEStatModel`, `MLEGradientBoostingModel`, `MLEHyperparameters`
- ✅ `minors/training.py` - `MLEModelTrainer`, `MLETrainingConfig`
- ✅ `minors/persistence.py` - `MLEModelStore`, `MLEModelMetadata`
- ✅ `tests/minors/test_model.py` - 19 tests for model classes
- ✅ `tests/minors/test_mle_training.py` - 7 tests for trainer
- ✅ `tests/minors/test_persistence.py` - 14 tests for persistence
- 117 total passing tests in minors module

**Implementation Notes**:
- `MLEStatModel` wraps LGBMRegressor with sample weight support for PA-weighted training
- `MLEGradientBoostingModel` contains one model per target stat (hr, so, bb, singles, doubles, triples, sb)
- Hyperparameters tuned for small dataset: `min_child_samples=20`, `max_depth=4`, `n_estimators=100`
- `MLEModelTrainer` uses `MLETrainingDataCollector` to gather training data with proper temporal alignment
- Support for validation set and early stopping during training
- `MLEModelStore` saves models to `~/.fantasy_baseball/models/mle/` with joblib serialization
- Model metadata includes training years, stats, feature names, and optional validation metrics
- Serialization roundtrip preserves prediction accuracy

### Phase 4: Integration ✅ COMPLETE

1. ✅ Implement `MLERateComputer` wrapper
2. ✅ Add `mle_pipeline()` preset
3. ✅ Update `PipelineBuilder` with `with_mle_rate_computer()` method
4. ✅ Integration tests with existing pipeline

**Deliverables**:
- ✅ `minors/rate_computer.py` - `MLERateComputer`, `MLERateComputerConfig`
- ✅ `pipeline/presets.py` - `mle_pipeline()` added to PIPELINES registry
- ✅ `pipeline/builder.py` - `with_mle_rate_computer()` method added
- ✅ `tests/minors/test_rate_computer.py` - 10 tests for rate computer

**Implementation Notes**:
- `MLERateComputer` implements the `RateComputer` protocol
- For players with `<mlb_pa_threshold` (default 200) MLB PA, MLE predictions are blended with Marcel rates
- Blending uses PA-weighted average: `(mlb_pa * marcel + milb_pa * mle) / total_pa`
- Requires MiLB stats from AAA or AA level with `min_milb_pa` (default 200) PA
- Falls back to Marcel for pitchers (MLE pitchers not yet implemented)
- Lazy-loads MLE model from `MLEModelStore` on first use
- Adds metadata to `PlayerRates`: `mle_applied`, `mle_source_level`, `mle_source_pa`, `marcel_rates`, `mle_rates`
- MiLB data cached via `CachedMinorLeagueDataSource` with appropriate TTLs

### Phase 5: Evaluation

1. Backtest on 2024 call-ups (held-out test set)
2. Compare to baselines (no MLE, traditional MLE)
3. Analyze failure modes
4. Document findings

**Deliverables**:
- Evaluation report
- Updated docs

### Phase 6: Pitcher Support (Optional, 1-2 weeks)

1. Extend all components for pitchers
2. Additional features (pitch mix, role)
3. Separate evaluation

---

## Open Questions

1. **Sample size threshold**: Should the model predict differently for 200 PA vs 500 PA MiLB seasons? (Yes—include PA as feature and let model learn appropriate weighting)

2. **Uncertainty quantification**: Should the model output confidence intervals? (Useful for blending decisions, but adds complexity—defer to v2)

3. **Pitcher handling**: Pitchers may need different features (K/9, BB/9, HR/9, GB%) and separate model training. Defer to Phase 6 or separate proposal.

## Resolved Questions

- **MiLB data source**: MLB Stats API provides full MiLB stats via `sportId` parameter. No scraping needed for basic stats.
- **Player ID mapping**: MLBAM IDs are consistent across MiLB and MLB in the Stats API.
- **Multi-level seasons**: PA-weighted aggregation across levels (see Training Data Construction).
- **Scouting features**: Omitted—sticking to pure statistical features for v1.
- **MiLB Statcast**: Available for AAA 2023+ via Baseball Savant. Optional enhancement for v2.

---

## References

- James, B. (1985). "Major League Equivalencies." *Baseball Abstract*.
- Silver, N. (2003). "Introducing PECOTA." *Baseball Prospectus*.
- Tango, T., Lichtman, M., Dolphin, A. (2007). *The Book: Playing the Percentages in Baseball*. Chapter on minor league translations.
- Zimmerman, J. (2019). "Minor League Translations in the Statcast Era." *FanGraphs*.
