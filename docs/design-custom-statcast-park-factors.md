# Custom Statcast Park Factors

## Overview

This document describes an approach to computing park factors from raw Statcast batted-ball data using `pybaseball`, rather than relying on pre-computed factors scraped from FanGraphs or Baseball Savant. The core idea is to compare actual outcomes (wOBAcon, BA, HR rate) against expected outcomes (xwOBA, xBA) at each venue. Because the expected metrics are derived from exit velocity and launch angle alone -- quantities that are independent of the park -- any systematic gap between actual and expected at a given venue isolates the pure park effect while controlling for batted-ball quality and roster talent.

This provider would conform to the existing `ParkFactorProvider` protocol defined in `src/fantasy_baseball_manager/pipeline/park_factors.py` and could serve as a drop-in replacement for or complement to `FanGraphsParkFactorProvider`.

## Data Sources

### Statcast pitch-level data

```python
from pybaseball import statcast

df = statcast(start_dt="2023-03-30", end_dt="2023-10-01")
```

Each row represents a single pitch. The subset of interest is batted-ball events, identified by a non-null `launch_speed`. Key columns:

| Column | Description |
|--------|-------------|
| `home_team` | Three-letter abbreviation of the home team (proxy for venue) |
| `launch_speed` | Exit velocity in mph |
| `launch_angle` | Launch angle in degrees |
| `hc_x`, `hc_y` | Hit coordinates on the field diagram |
| `hit_distance_sc` | Estimated distance of the batted ball |
| `events` | Outcome of the plate appearance (single, double, home_run, field_out, etc.) |
| `stand` | Batter handedness (L or R) |
| `p_throws` | Pitcher handedness (L or R) |
| `estimated_ba_using_speedangle` | Statcast xBA for the batted ball |
| `estimated_woba_using_speedangle` | Statcast xwOBA for the batted ball |

### Spray angle

```python
from pybaseball.datahelpers.statcast_utils import add_spray_angle

df = add_spray_angle(df, adjusted=True)
```

This adds a `spray_angle` column (in degrees) that can be used to bucket batted balls into pull, center, and opposite-field zones. The `adjusted=True` flag normalizes for batter handedness so that positive values always represent the pull side.

## Methodology

### Step 1: Collect batted-ball events

Pull all Statcast data for the desired window (recommended: 3 full seasons) and filter to rows where `launch_speed` is not null. This yields roughly 120,000--130,000 batted-ball events per season.

```python
bb = df[df["launch_speed"].notna()].copy()
```

### Step 2: Add spray angles

Attach spray angle data for optional directional splits.

```python
bb = add_spray_angle(bb, adjusted=True)
```

### Step 3: Bucket by batted-ball characteristics

Group batted balls into zones defined by exit velocity, launch angle, and spray angle. This bucketing is used for the HR-rate and hit-type comparisons in Step 5, ensuring that the park comparison controls for the quality and direction of contact.

Suggested zone boundaries:

- **Exit velocity**: soft (<85 mph), medium (85--95 mph), hard (95--105 mph), barrel (105+ mph)
- **Launch angle**: ground ball (<10 deg), low line drive (10--18 deg), line drive (18--28 deg), fly ball (28--40 deg), high fly ball / pop-up (>40 deg)
- **Spray angle** (adjusted): pull (<-15 deg), center (-15 to 15 deg), opposite (>15 deg)

These thresholds can be tuned. Finer granularity improves accuracy but requires more data for stability.

### Step 4: Compute actual vs. expected by park

For each park (keyed on `home_team`), compute:

- **wOBAcon park factor**: `mean(woba_value) / mean(estimated_woba_using_speedangle)` across all batted balls at that venue.
- **BA park factor**: `mean(hit_flag) / mean(estimated_ba_using_speedangle)` where `hit_flag` is 1 for singles, doubles, triples, and home runs, 0 otherwise.

A factor greater than 1.0 indicates the park inflates that stat; less than 1.0 indicates suppression.

### Step 5: Per-stat factors

Compute separate factors for individual outcome types:

- **HR factor**: Among fly balls (launch angle 25--40 deg), compare the actual HR rate to the expected HR rate. The expected HR rate can be derived from the xwOBA model or from a separate EV/LA lookup table of league-average HR probability.
- **1B / 2B / 3B factors**: Among batted balls that are hits (or expected hits by xBA), compare the distribution of hit types at the venue to the league-wide distribution for the same EV/LA/spray-angle bucket.
- **BB / SO factors**: Walk and strikeout rates are less park-dependent but not zero. Altitude affects pitch movement and perceived velocity, so Colorado and other elevated parks may show elevated strikeout or walk differentials. These can be computed as simple rate ratios (home games vs. road games for teams playing at that venue) and regressed heavily toward 1.0.

### Step 6: Handedness splits

Repeat Step 4 separately for left-handed and right-handed batters (using the `stand` column). This produces L/R park factor splits, which are valuable for projecting platoon-heavy lineups.

### Step 7: Spray-direction factors

Repeat Step 4 separately for pull, center, and opposite-field contact. Some parks disproportionately affect pulled fly balls (short porches) while being neutral on opposite-field contact. These directional factors can refine projections for pull-heavy or spray-oriented hitters.

### Step 8: Regression toward 1.0

Raw park factors are noisy, especially for less common events (triples, for example). Apply Bayesian regression toward 1.0:

```
regressed_factor = (n * raw_factor + k * 1.0) / (n + k)
```

Where `n` is the number of relevant batted-ball events at the park and `k` is a regression constant representing the "prior" sample size. Suggested values for `k`:

| Stat | Regression constant (k) |
|------|------------------------|
| wOBAcon | 3000 |
| BA | 3000 |
| HR | 1500 |
| 2B | 2000 |
| 3B | 500 |
| 1B | 2500 |
| BB | 5000 |
| SO | 5000 |

Lower constants for HR and 3B reflect the fact that these events have higher signal-to-noise ratios for park effects (park geometry directly determines whether a fly ball leaves the yard or a ball rolls to the wall). Higher constants for BB and SO reflect their weak relationship to park geometry.

When averaging across multiple seasons, weight more recent seasons more heavily (e.g., 5/3/2 weighting for the most recent three years).

## Per-Stat Factor Summary

| Stat | Method | Park-sensitivity |
|------|--------|-----------------|
| HR | Actual HR rate on fly balls vs. expected HR rate by EV/LA | High |
| 2B | Actual 2B rate vs. expected by EV/LA/spray | Moderate |
| 3B | Actual 3B rate vs. expected by EV/LA/spray | High (outfield dimensions, surface) |
| 1B | Actual 1B rate vs. xBA-derived expectation | Low--moderate |
| BB | Home/road rate ratio, heavily regressed | Low |
| SO | Home/road rate ratio, heavily regressed | Low |
| wOBAcon | Actual wOBAcon vs. xwOBAcon | Moderate--high |

## Advantages

- **Full control over smoothing, regression, and granularity.** Unlike scraped factors, every parameter is tunable and transparent.
- **Handedness and spray-direction splits.** FanGraphs publishes basic L/R splits but not spray-direction factors. This approach supports arbitrary splits.
- **Controls for batted-ball quality.** By comparing actual vs. xwOBA/xBA, roster talent is factored out. A team of sluggers playing in a neutral park will not inflate that park's factor.
- **Stat-level granularity.** Factors can be computed for any outcome type or custom batted-ball bucket (e.g., "barrel rate on pulled fly balls").
- **Reproducibility.** The computation is deterministic given the same input data, unlike factors scraped from third-party sites that may change methodology between seasons.

## Challenges

### Data volume

A full season of Statcast data contains roughly 700,000+ pitches. The batted-ball subset is approximately 120,000--130,000 rows per season. Three seasons of data is around 400,000 batted-ball events. This is manageable in memory but too slow to recompute on every run.

### Processing time

Downloading three seasons of Statcast data via `pybaseball.statcast()` takes several minutes per season due to the Baseball Savant API rate limits. The computation itself (groupby, aggregation) is fast once the data is in memory.

### Sample size

Each park sees roughly 4,000--4,500 batted-ball events per season. Over three seasons, that is roughly 12,000--13,500 events per park, which is sufficient for overall wOBAcon and BA factors but thin for triple-split (handedness x spray-direction x stat) combinations. Aggressive regression is necessary for fine-grained splits.

### Edge cases

- **New stadiums**: A park with only one season of data should be regressed more heavily. The regression constant `k` can be scaled inversely with the number of available seasons.
- **Mid-season park changes**: Rare but possible (e.g., wall height modifications). If detected, treat pre- and post-change data as separate parks or exclude the transition season.
- **Retractable roofs**: Parks like Chase Field and Minute Maid Park behave differently with the roof open vs. closed. Statcast does not reliably encode roof status. Consider ignoring this distinction and accepting a blended factor, or supplementing with weather data if available.
- **Team relocation or rebranding**: The `home_team` abbreviation may change. Maintain a mapping of historical abbreviations to current ones.

### Statcast data quality

Statcast tracking accuracy varies by park and has improved over time. Early seasons (2015--2016) have higher rates of missing `launch_speed` and `launch_angle` data. Recommend using 2020 onward for the most reliable data, excluding the shortened 2020 season or weighting it proportionally.

## Integration

### Implementation

Create a new `StatcastParkFactorProvider` class conforming to the `ParkFactorProvider` protocol:

```python
class StatcastParkFactorProvider:
    def __init__(
        self,
        *,
        years_to_average: int = 3,
        regression_constants: dict[str, float] | None = None,
        season_weights: list[float] | None = None,
        handedness_splits: bool = False,
    ) -> None:
        ...

    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        ...
```

When `handedness_splits` is enabled, the returned stat keys would include a suffix (e.g., `hr_L`, `hr_R`, `woba_L`, `woba_R`).

### Caching

Wrap `StatcastParkFactorProvider` with the existing `CachedParkFactorProvider`, which persists computed factors in SQLite via `CacheStore`. Raw Statcast data downloads should also be cached separately (the data does not change once a season is complete), potentially as Parquet files on disk to avoid repeated multi-minute downloads.

### Usage

The provider can be used as a direct replacement for `FanGraphsParkFactorProvider`:

```python
provider = CachedParkFactorProvider(
    delegate=StatcastParkFactorProvider(years_to_average=3),
    cache=sqlite_cache,
)
factors = provider.park_factors(year=2025)
```

Or both providers can be composed (e.g., averaging FanGraphs and Statcast factors) for a blended approach.

## References

- Chamberlain, A. "2019 Statcast Park Factors and the Importance of Spray Angle." FanGraphs Community Research, 2019.
- Baseball Prospectus. "Statcast Exit Velocity: A Statistical Assessment of Park Effects."
- Tango, T., Lichtman, M., Dolphin, A. *The Book: Playing the Percentages in Baseball.* Potomac Books, 2007. Chapter on park factors.
- Lichtman, M. "UZR and Park Factors." The Book Blog. Discussion of how park geometry affects defensive metrics and, by extension, BABIP-based park factors.
