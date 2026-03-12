# SGP Overhaul Roadmap

SGP (Standings Gain Points) was built as a principled alternative to ZAR — using empirical standings data to weight categories by competitive value rather than statistical variance. Despite this theoretical advantage, SGP lost decisively to ZAR-reformed in holdout validation (WAR ρ 0.063 vs 0.216 in 2024, pitcher WAR ρ near zero). The FanGraphs "Great Valuation System Test" crowned SGP the winner by a wide margin, so the methodology isn't flawed — our implementation is.

This roadmap addresses four specific implementation gaps identified by comparing our system against the published methodology from Smart Fantasy Baseball, the FanGraphs test, and Todd Zola's SGP Theory:

1. **Mean-gap denominators** — our computation averages adjacent-team gaps, which is algebraically equivalent to `(1st - last) / (n-1)` and is dominated by outlier teams. The literature strongly favors regression-slope denominators.
2. **Ad-hoc volume weighting** — our rate-stat formula multiplies by `player_vol / avg_vol`, a linear approximation. The correct approach computes each player's marginal impact on a representative team's aggregate rate stat.
3. **SV+HLD still active** — the category that's ρ = -0.608 against WAR was never removed from SGP, despite being the primary fix that made ZAR-reformed succeed.
4. **Redraft denominators applied to keeper league** — league size and roster structure affect standings gaps, and our denominator source doesn't match the target league.

Each fix is independently testable. The final phase runs a full holdout comparison against ZAR-reformed to determine if SGP can become competitive.

## Status

| Phase | Status |
|-------|--------|
| 1 — Regression-slope denominators | not started |
| 2 — Team-impact rate-stat formula | not started |
| 3 — Category signal alignment | not started |
| 4 — Holdout validation | not started |

## Phase 1: Regression-slope denominators

Replace the mean-gap denominator computation with linear-regression slope, matching the methodology that won the FanGraphs valuation system test.

### Context

The current `compute_sgp_denominators()` in `services/sgp_denominator.py` sorts teams by category total, computes gaps between adjacent teams, and takes the mean. This is Art McGee's original 1997 method. It has a known weakness: it reduces to `(first_place_value - last_place_value) / (n_teams - 1)`, so a single tanking team or runaway leader can distort the denominator for an entire season.

The regression-slope method fits a line through `(standings_points, category_total)` for all teams and returns the slope. This uses all data points equally, is robust to outliers, and was the method used by the winning system in the FanGraphs test. Tanner Bell's [Improved SGP Calculation Formula](https://www.smartfantasybaseball.com/2014/04/improved-sgp-calculation-formula-part-1/) demonstrates the difference with worked examples.

The regression slope answers the same question — "how many units of this stat are needed to gain one standings point?" — but estimates it more accurately.

### Steps

1. **Add a `method` parameter to `compute_sgp_denominators()`** accepting `"mean_gap"` (current default) or `"regression"`. When `"regression"`:
   - For each category and season, assign standings points 1..N based on rank in that category (1 = worst, N = best).
   - Compute the slope of the linear regression `category_total ~ standings_points` using `numpy` or `statistics` (slope = Σ[(x-x̄)(y-ȳ)] / Σ[(x-x̄)²]).
   - The slope is the SGP denominator: how many stat units per one standings point.
   - For "lower is better" categories (ERA, WHIP), negate as before.
2. **Handle ties in standings points.** If two teams have identical category totals, they should share the standings points (mean rank). This matters for small leagues.
3. **Update the `fbm sgp denominators` CLI command** to accept a `--method` flag and display both methods side by side for comparison.
4. **Add tests.** Synthetic standings data where mean-gap and regression produce different results (include an outlier team). Verify regression is more stable. Verify backward compatibility — `method="mean_gap"` produces identical output to current implementation.
5. **Compare denominators.** Run both methods on the actual 14 seasons of redraft standings. Document the per-category differences and coefficient of variation for each method.

### Acceptance criteria

- Regression-slope denominators computed for all 10 categories across all available seasons.
- Regression denominators have lower coefficient of variation than mean-gap denominators across seasons (expected based on the literature).
- `method="mean_gap"` produces identical output to current implementation (backward compatible).
- CLI shows both methods for easy comparison.

---

## Phase 2: Team-impact rate-stat formula

Replace the volume-weighted multiplier with the correct marginal team-impact formula for rate stats.

### Context

The current volume-weighted approach (`raw_sgp * player_vol / avg_vol`) is a linear approximation of the real question: "how much does adding this player to a representative team move that team's aggregate rate stat?" The correct formula, described in [Smart Fantasy Baseball](https://www.smartfantasybaseball.com/2018/02/more-than-you-wanted-to-know-about-ratio-stats-and-standings-gain-points/), computes this directly:

```
team_rate_with    = (team_numerator + player_numerator) / (team_denominator + player_denominator)
team_rate_without = team_numerator / team_denominator
marginal_impact   = team_rate_with - team_rate_without   # (or without - with, for lower-is-better)
rate_sgp          = marginal_impact / sgp_denominator
```

Where `team_numerator` and `team_denominator` are league-average team totals (e.g., ~475 ER and ~1200 IP for ERA, derived from standings data or projection aggregates).

This formula:
- Naturally incorporates volume (a 200 IP pitcher shifts the denominator more than a 60 IP pitcher)
- Handles nonlinearity (adding 200 IP to a 1200 IP base is a bigger share than the linear approximation assumes)
- Uses the same "representative team" concept as the SGP denominator itself, making the system internally consistent
- Eliminates the need for the `volume_weighted` flag and `avg_volumes` computation

### Steps

1. **Compute representative team totals.** For each rate-stat category, derive the league-average team numerator and denominator from standings data (preferred) or from projections (fallback). Store these alongside SGP denominators — they're a natural companion. For example, `team_er = mean(team_totals["er"])`, `team_ip = mean(team_totals["ip"])` across recent seasons.
2. **Add a `team_impact` scoring mode to `compute_sgp_scores()`.** When enabled, replace the current rate-stat formula with the marginal team-impact formula. Accept representative team totals as an additional parameter.
3. **Remove the `volume_weighted` flag** (or deprecate it). The team-impact formula subsumes it — there's no reason to keep both. The `volume_weighted=False` (unweighted) path should remain for backward-compatible testing but can be marked as legacy.
4. **Update `SgpModel`** to compute or load representative team totals and pass them to the pipeline.
5. **Add tests.**
   - Two pitchers with identical ERA but different IP: verify SGP scales with IP (as before, but via the team-impact formula rather than a multiplier).
   - Compare team-impact SGP to volume-weighted SGP on the same data: values should be similar but not identical (nonlinearity).
   - Edge case: pitcher with 0 IP gets SGP = 0.
   - Verify that counting stats are completely unaffected.

### Acceptance criteria

- Rate-stat SGP is computed via the marginal team-impact formula using representative team totals.
- Two pitchers with identical ERA but IP of 180 vs 60 get different SGP scores that reflect their actual team impact (not a simple 3:1 ratio — the nonlinearity should be visible).
- Representative team totals are derived from standings data (same source as denominators).
- Counting-stat SGP is unchanged.
- Existing volume-weighted tests continue to pass with the legacy flag, and new tests validate the team-impact path.

---

## Phase 3: Category signal alignment

Apply the same category-signal fixes that made ZAR-reformed successful: drop SV+HLD and raise the pitcher eligibility floor.

### Context

The valuation-reform roadmap proved that removing SV+HLD and raising min_ip from 30 to 60 are the two highest-impact changes for pitcher accuracy (pitcher WAR ρ improved by +0.069 in 2024, +0.064 in 2025). SGP was tested in that validation *without* these fixes — it still included SV+HLD and used the league's default min_ip of 30. This is an apples-to-oranges comparison: ZAR-reformed benefited from signal cleanup that SGP never received.

This phase creates an `sgp-reformed` variant that mirrors zar-reformed's signal choices, establishing a fair baseline for the phase 4 comparison.

### Steps

1. **Create `sgp-reformed` model variant** in `models/sgp_reformed/`. Follow the same pattern as `zar-reformed`: wrap `SgpModel` with parameter overrides:
   - `category_weights = {"sv+hld": 0.0}` — but SGP doesn't currently support category weights. Add an optional `category_weights` parameter to `compute_sgp_scores()` that multiplies each category's SGP before summing the composite, identical to how ZAR handles it.
   - Override min_ip to 60 (read from model_params, same as zar-reformed).
2. **Add category weights to `compute_sgp_scores()`**. This is a small change — multiply `category_sgp[cat.key]` by `category_weights.get(cat.key, 1.0)` before summing composite. Reuse the same semantics as ZAR: `None` means all weights = 1.0, weight = 0.0 removes the category.
3. **Register `sgp-reformed` in the model registry.** It should be invocable via `fbm predict sgp-reformed`.
4. **Add tests.** Verify category weights work. Verify weight=0 removes a category from composite. Verify min_ip override filters the pitcher pool.

### Acceptance criteria

- `sgp-reformed` produces valuations with SV+HLD removed and min_ip=60.
- `sgp` (baseline) produces identical output to current production (backward compatible).
- Category weights are applied to SGP composite scores using the same semantics as ZAR.
- `fbm predict sgp-reformed` is a registered, runnable command.

---

## Phase 4: Holdout validation

Run the full comparison matrix: baseline SGP, SGP with regression denominators, SGP with team-impact rates, and the fully reformed SGP — all against ZAR-reformed.

### Context

Each prior phase produces an independently testable improvement. This phase measures their individual and combined effects on holdout accuracy and determines whether the overhauled SGP can compete with ZAR-reformed as a production system.

The evaluation uses established independent targets (WAR ρ, top-N hit rates) via `fbm valuations compare --check`. The bar is ZAR-reformed's holdout performance: WAR ρ 0.216/0.241 (2024/2025), pitcher WAR ρ 0.218/0.248.

### Steps

1. **Define the test grid.** At minimum, evaluate these configurations on seasons 2024 and 2025:

   | Config | Denominators | Rate Stats | SV+HLD | min_ip |
   |--------|-------------|------------|--------|--------|
   | sgp-baseline | mean-gap | unweighted | active | 30 |
   | sgp-regression | regression | unweighted | active | 30 |
   | sgp-team-impact | mean-gap | team-impact | active | 30 |
   | sgp-reformed | regression | team-impact | removed | 60 |
   | zar-reformed (control) | n/a | n/a | removed | 60 |

2. **Generate holdout valuations** for each configuration using ensemble projections.
3. **Run `fbm valuations compare`** for each candidate against zar-reformed on both seasons. Record WAR ρ (overall, batter, pitcher), top-25/50/100 hit rates.
4. **Build the comparison matrix.** Identify which fixes matter most. Expected rank order of impact: SV+HLD removal > team-impact rates > regression denominators (based on the prior valuation-reform results and the magnitude of each issue).
5. **Pitcher-focused inspection.** For `sgp-reformed`: are the top-20 pitcher valuations plausible? Are reliever distortions eliminated? Are SP/RP rankings sensible?
6. **Run regression gate checks** (`--check`) for `sgp-reformed` vs `zar-reformed` on both seasons.
7. **Adopt, blend, or reject:**
   - **Adopt** if `sgp-reformed` matches or beats `zar-reformed` on all independent targets.
   - **Blend** if SGP and ZAR have complementary strengths (e.g., SGP better on pitchers, ZAR better on batters) — consider an ensemble valuation.
   - **Reject** if `sgp-reformed` still underperforms ZAR-reformed, documenting which fixes helped and which didn't.
8. **Document results** in this roadmap's status table with the full comparison matrix.

### Acceptance criteria

- Comparison matrix covering all configurations on both holdout seasons with WAR ρ and hit-rate metrics.
- Each phase's individual contribution is measured (regression denominators alone, team-impact rates alone, SV+HLD removal alone).
- `sgp-reformed` passes `--check` against `zar-reformed` on at least one holdout season (WAR ρ not more than 0.01 worse, hit rates not more than 5pp worse).
- Pitcher WAR ρ for `sgp-reformed` exceeds baseline SGP's pitcher WAR ρ on both seasons (expected: large improvement from near-zero baseline).
- Go/no-go decision documented with supporting data.

### Gate: go/no-go

**Go (adopt)** if `sgp-reformed` matches or improves ZAR-reformed's WAR ρ on both seasons without degrading hit rates. **Go (blend)** if `sgp-reformed` improves pitcher metrics but slightly degrades batter metrics — pursue an ensemble approach. **No-go** if `sgp-reformed` still underperforms ZAR-reformed on overall WAR ρ on both seasons despite all fixes. In that case, SGP's weakness may be fundamental to H2H category leagues (SGP was designed for and validated on roto, not H2H), and further investment should be redirected to [ZAR Pitcher Budget](zar-pitcher-budget.md) improvements instead.

---

## Ordering

- **Phases 1, 2, and 3** are independent and can be implemented in any order or in parallel. Phase 1 changes denominator computation, phase 2 changes rate-stat scoring, phase 3 adds category weights and creates the reformed variant. None modify shared code paths.
- **Phase 4** depends on all prior phases (needs all configurations available for the grid search).
- This roadmap is independent of [ZAR Pitcher Budget](zar-pitcher-budget.md) — they address the same symptom (pitcher overvaluation) from different angles. If both produce improvements, the better system wins; if both improve different aspects, an ensemble is possible.
- The [SGP Rate-Stat Volume Weighting](sgp-rate-stat-volume-weighting.md) roadmap's phase 2 (deferred holdout validation) is superseded by this roadmap's phase 4, which tests a strictly better rate-stat approach (team-impact formula subsumes volume weighting).
