# Design: Pitching Park Factor Adjustments

## Overview

Extend park factor adjustments to pitching projections. The `ParkFactorAdjuster` stage
currently only handles batting rate stats (`hr`, `singles`, `doubles`, `triples`, `bb`,
`so`), leaving pitcher rates unadjusted. This creates an asymmetry that actively hurts
pitching projection accuracy -- evaluation showed that enabling park factors dropped
pitching Top-20 precision from 0.317 to 0.300.

## Problem Statement

- `ParkFactorAdjuster.adjust()` divides each rate by its corresponding park factor, but
  only batting stat names (`singles`, `doubles`, `triples`, `hr`, `bb`, `so`) match the
  keys returned by `ParkFactorProvider`. Pitching rates use different stat names (`h`,
  `bb`, `so`, `hr`, `er`, `hbp`, `w`, `sv`, `hld`, `bs`) so most pass through with the
  default factor of 1.0.
- Pitchers in hitter-friendly parks (Coors, Great American) carry inflated HR, H, BB,
  and ER rates that make their projections appear worse than true talent.
- Pitchers in pitcher-friendly parks (Oracle, Petco) carry suppressed rates that make
  their projections appear better than true talent.
- The asymmetry is counterproductive: batting rates get neutralized while pitching rates
  do not, so any downstream interaction (e.g., valuation ranking across positions)
  introduces a systematic bias.

### Why only `bb`, `so`, and `hr` match today

The `FanGraphsParkFactorProvider._column_map()` returns keys `hr`, `singles`, `doubles`,
`triples`, `bb`, `so`. Of these, `bb`, `so`, and `hr` happen to overlap with pitching
component stat names, so those three pitching rates already receive a partial adjustment.
However, the critical pitching stats `h` (hits) and `er` (earned runs) have no matching
park factor key and always pass through at 1.0. This partial overlap is the root cause of
the precision regression -- it perturbs some rates but not the ones that feed most
directly into ERA and WHIP.

## Proposed Approach

### Stat Mapping

Map existing park factor stat names to pitching rate stat names. The
`ParkFactorProvider` returns factors keyed by batting stat names. The adjuster needs a
translation layer to apply those same factors to the corresponding pitching rates.

| Park Factor Key | Pitching Rate Stat | Derivation |
|---|---|---|
| `hr` | `hr` | Direct match -- already works today |
| `bb` | `bb` | Direct match -- already works today |
| `so` | `so` | Direct match -- already works today |
| `singles` | -- | No direct pitching equivalent (see composite `h` below) |
| `doubles` | -- | No direct pitching equivalent (see composite `h` below) |
| `triples` | -- | No direct pitching equivalent (see composite `h` below) |
| (composite) | `h` | Derived from `singles`, `doubles`, `triples`, and `hr` factors |
| (composite) | `er` | Derived from `h`, `bb`, and `hr` factors combined |

#### Deriving the hits (`h`) factor

Pitching `h` encompasses all hit types. A composite hits factor can be computed as a
weighted average of the individual hit-type factors:

```
h_factor = (w_1b * f_1b + w_2b * f_2b + w_3b * f_3b + w_hr * f_hr)
         / (w_1b + w_2b + w_3b + w_hr)
```

where `w_*` are league-average frequencies of each hit type and `f_*` are the
corresponding park factors. As a practical simplification, since singles dominate hit
composition (~60-65% of all hits), a reasonable approximation is:

```
h_factor = 0.63 * f_1b + 0.19 * f_2b + 0.04 * f_3b + 0.14 * f_hr
```

These weights can be derived from actual league averages already computed by
`compute_pitching_league_rates()` or hardcoded from recent league-average distributions.

#### Deriving the earned runs (`er`) factor

Earned runs are a downstream consequence of hits, walks, and home runs. Rather than
adjusting `er` directly (which risks double-counting with the component adjustments),
there are two options:

1. **Adjust components only.** Adjust `h`, `bb`, `hr`, and `so` rates, then let `er`
   remain unadjusted. Since ERA is computed in `StandardFinalizer.finalize_pitching()`
   from `er` and `ip`, the `er` rate would still carry park effects. This leaves money
   on the table.

2. **Adjust `er` with its own composite factor.** Compute an earned-run factor as a
   weighted blend of `h_factor`, `bb` factor, and `hr` factor. A simple approach:

   ```
   er_factor = 0.50 * h_factor + 0.25 * bb_factor + 0.25 * hr_factor
   ```

   The weights reflect the relative contribution of each component to run scoring.
   This is imprecise but directionally correct and avoids the need to re-derive ERA
   from adjusted components.

**Recommendation:** Option 2 -- adjust `er` directly with a composite factor. This is
simpler to implement and avoids restructuring the finalizer. The composite weights can be
tuned empirically by evaluating precision on held-out seasons.

### Adjustment Direction

The adjustment formula is the same as for batting: `neutralized_rate = rate / factor`.

Example: A pitcher with an `h` rate of 0.30 (per out) in Coors Field where the
composite hits factor is 1.10 has a neutralized `h` rate of 0.30 / 1.10 = 0.273.

A pitcher with an `hr` rate of 0.035 in Coors (HR factor 1.15) has a neutralized `hr`
rate of 0.035 / 1.15 = 0.030.

### Implementation Options

#### Option A: Extend `ParkFactorAdjuster` with a stat-name mapping config

Add a mapping parameter to `ParkFactorAdjuster.__init__()` that translates park factor
keys to player rate stat names. The adjuster would accept a
`stat_mapping: dict[str, str] | None` parameter. When set, each park factor key is
looked up through the mapping before matching against player rate stat names.

For batting, no mapping is needed (stat names already match). For pitching, the mapping
would handle the `h` and `er` composite derivations.

```python
class ParkFactorAdjuster:
    def __init__(
        self,
        provider: ParkFactorProvider,
        pitching_stat_mapping: dict[str, str | Callable] | None = None,
    ) -> None:
        ...
```

**Pros:** Single adjuster class, no duplication.
**Cons:** The composite factor logic (`h`, `er`) makes a simple key mapping insufficient.
The adjuster needs to compute derived factors, which adds complexity to a currently clean
class.

#### Option B: Create a `PitchingParkFactorAdjuster` stage

A separate stage that handles pitching-specific mappings, including composite factor
derivation. It would use the same `ParkFactorProvider` but transform the raw factors
into a pitching-specific factor dict before applying them.

```python
class PitchingParkFactorAdjuster:
    def __init__(self, provider: ParkFactorProvider) -> None:
        ...

    def _compute_pitching_factors(
        self, raw: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Derive pitching-specific factors from batting park factors."""
        ...

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        ...
```

**Pros:** Clean separation of concerns. Pitching factor derivation logic is isolated.
Batting adjuster remains unchanged. Easy to test independently.
**Cons:** Two adjuster stages in the pipeline for park-factor-enabled presets. Slightly
more wiring in `presets.py`.

#### Option C: Unified adjuster with player-type detection

Have the adjuster inspect `PlayerRates.metadata` (e.g., check for an `is_starter` key
or a `player_type` field) to determine whether it is processing a batter or pitcher, and
apply the appropriate factor mapping.

**Pros:** Single stage in the pipeline, automatic dispatch.
**Cons:** Relies on metadata conventions. Couples the adjuster to pipeline-specific
metadata keys. Harder to test. Fragile if metadata changes.

### Recommendation

**Option B** -- create a separate `PitchingParkFactorAdjuster`. The composite factor
derivation for `h` and `er` is pitching-specific logic that does not belong in the
batting adjuster. The pipeline engine already calls `project_batters()` and
`project_pitchers()` through separate code paths (both call the same adjuster chain, but
only one path processes pitching `PlayerRates`). A pitching-specific stage that is a
no-op for batting data (or vice versa) fits naturally.

Since `project_batters()` and `project_pitchers()` share the same `adjusters` tuple, the
pitching adjuster must be safe to call on batting `PlayerRates` (and the existing batting
adjuster must be safe on pitching `PlayerRates`). Today the batting adjuster is already
safe on pitching data -- unmatched stat names fall through to factor 1.0. The new
pitching adjuster should similarly be a no-op for batting data, which can be achieved by
checking for a metadata key like `is_starter` or by only mapping stats that exist in
pitching rate dicts (e.g., `h` and `er` are not present in batting rates).

## Pitching-Specific Considerations

### ERA derivation path

ERA is computed in `StandardFinalizer.finalize_pitching()` as `(er / ip) * 9`. The `er`
rate flows through the pipeline as a per-out rate. Adjusting `er` directly with a
composite park factor is the most straightforward way to improve ERA projections without
changing the finalizer.

### WHIP derivation path

WHIP is computed as `(h + bb) / ip`. If both `h` and `bb` rates are adjusted by their
respective park factors, WHIP will be correctly neutralized without any additional logic.

### FIP-like components

HR, BB, and K rates map cleanly to individual park factors (`hr`, `bb`, `so`). These
already partially work today because the stat names overlap. The main gap is `h` and
`er`.

### Pipeline stage ordering

The current adjuster ordering in park-factor-enabled presets is:

1. `ParkFactorAdjuster` -- neutralize park effects
2. `RebaselineAdjuster` -- scale to target-year league environment
3. `MarcelAgingAdjuster` -- apply age curves

The pitching adjuster should run at the same position (before rebaselining), either as a
second adjuster at position 1.5 or by replacing the single `ParkFactorAdjuster` with
both stages:

```python
adjusters=(
    ParkFactorAdjuster(_cached_park_factor_provider()),
    PitchingParkFactorAdjuster(_cached_park_factor_provider()),
    RebaselineAdjuster(),
    MarcelAgingAdjuster(),
)
```

Both adjusters can share the same `CachedParkFactorProvider` instance to avoid redundant
fetches.

## Stat Name Mapping Table

Summary of all mappings from park factor keys to pitching rate stats:

| Park Factor Key | Pitching Rate Stat | Type | Notes |
|---|---|---|---|
| `hr` | `hr` | Direct | Most impactful single mapping. Already works. |
| `bb` | `bb` | Direct | Altitude and park dimensions affect pitch movement. Already works. |
| `so` | `so` | Direct | High-K parks inflate pitcher K rate. Already works. |
| `singles` | `h` (partial) | Composite | Combined with doubles, triples, hr for composite hits factor. |
| `doubles` | `h` (partial) | Composite | See above. |
| `triples` | `h` (partial) | Composite | See above. |
| `hr` | `er` (partial) | Composite | HR, hits, and BB factors blended for earned-run factor. |
| `bb` | `er` (partial) | Composite | See above. |
| (composite `h`) | `er` (partial) | Composite | See above. |

## Risks

- **Noisy park factors for SO/BB.** Strikeout and walk park factors have higher
  year-to-year variance than HR factors. The existing `FanGraphsParkFactorProvider`
  already mitigates this with multi-year averaging and regression toward 1.0, but the
  composite `h` and `er` factors add another layer of estimation noise.

- **Double-counting.** If ERA or WHIP were ever adjusted directly *and* their component
  rates were also adjusted, the correction would be applied twice. The current design
  avoids this because ERA and WHIP are derived stats computed in the finalizer, not
  stored as rates in `PlayerRates`. As long as the adjuster only touches `h`, `bb`,
  `hr`, `so`, and `er` rates, there is no double-counting.

- **Composite weight sensitivity.** The weights used to derive `h_factor` and `er_factor`
  are approximations. If the weights are significantly wrong, the adjustment could
  introduce bias rather than remove it. Mitigation: tune weights against evaluation
  results across multiple seasons.

- **Stage ordering dependency.** The pitching adjuster must run before `RebaselineAdjuster`
  to avoid distorting the rebaseline ratios. This is already the natural position in the
  pipeline, but it should be documented as a requirement.

- **Shared adjuster chain.** Since `project_batters()` and `project_pitchers()` use the
  same `adjusters` tuple, the pitching adjuster will be called on batting `PlayerRates`
  (and vice versa). Both adjusters must be no-ops for the wrong player type. This is
  achievable but must be tested.

## Integration

1. **Create `PitchingParkFactorAdjuster`** in
   `src/fantasy_baseball_manager/pipeline/stages/pitching_park_factor_adjuster.py`.
   - Accept a `ParkFactorProvider` via constructor injection.
   - Implement `adjust(players: list[PlayerRates]) -> list[PlayerRates]`.
   - Derive composite `h` and `er` factors from the raw park factor dict.
   - Apply adjustments only to pitching-specific stats (`h`, `er`). Skip if the
     player's rates do not contain these keys (i.e., it is a batter).

2. **Wire into presets** in `src/fantasy_baseball_manager/pipeline/presets.py`.
   - Add `PitchingParkFactorAdjuster` to `marcel_park_pipeline()`,
     `marcel_plus_pipeline()`, and `marcel_full_pipeline()`.
   - Place it immediately after `ParkFactorAdjuster` in the `adjusters` tuple.
   - Share the same `_cached_park_factor_provider()` instance.

3. **Add tests** in `tests/pipeline/stages/test_pitching_park_factor_adjuster.py`.
   - Test composite factor derivation (known inputs -> expected `h` and `er` factors).
   - Test that batting `PlayerRates` pass through unchanged.
   - Test that pitching `PlayerRates` are adjusted correctly.
   - Test edge cases: missing team, missing park factor data, zero factors.

4. **Re-run evaluation** to confirm pitching Top-20 precision improves (target: at least
   back to 0.317 baseline, ideally higher) and batting precision is not regressed.
