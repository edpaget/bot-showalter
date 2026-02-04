# Minor League Equivalencies (MLE)

## What

Translate minor league performance into major league equivalent stats using level-specific translation factors, enabling projections for players with little or no MLB track record.

## Why

The current Marcel system requires MLB history, leaving rookies and call-ups with no projection or a heavily regressed one based on minimal MLB data. MLE translations provide a principled way to incorporate minor league performance, which is the best available signal for these players. This is particularly valuable for fantasy drafts where rookie upside is a key differentiator.

## Pipeline Fit

New `MLERateComputer` that wraps the base `RateComputer`. For players with insufficient MLB history (< 200 PA), it substitutes translated minor league rates. For players with partial MLB history, it blends MLB and MLE rates weighted by sample size.

## Data Requirements

- Minor league stats by level (pybaseball's `minor_league_stats()` or custom scraper)
- Translation factors per level (AAA, AA, A+, A, etc.) and per league
- Historical calibration data to derive/validate translation factors

## Key References

- James, B. "Major League Equivalencies" (original concept, 1985)
- Silver, N. "PECOTA Minor League Equivalencies" (Baseball Prospectus methodology)
- Tango, T. "Marcel Minor League Projections" (The Book Blog)

## Implementation Sketch

1. Define translation factor tables per level and stat:
   - AAA -> MLB: ~0.90 for rate stats
   - AA -> MLB: ~0.80
   - A+ -> MLB: ~0.70
   - A -> MLB: ~0.60
2. Create `MinorLeagueDataSource` protocol for fetching MiLB stats
3. Translate: `mlb_equivalent_rate = minor_rate * translation_factor`
4. Apply standard regression with extra PA/outs of regression for translation uncertainty
5. Blend with any available MLB data: `rate = (mlb_pa * mlb_rate + mle_pa * mle_rate) / (mlb_pa + mle_pa)`
