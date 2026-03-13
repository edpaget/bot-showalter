# SGP Overhaul Roadmap

SGP (Standings Gain Points) was built as a principled alternative to ZAR — using empirical standings data to weight categories by competitive value rather than statistical variance. Despite this theoretical advantage, SGP lost decisively to ZAR-reformed in holdout validation (WAR ρ 0.063 vs 0.216 in 2024, pitcher WAR ρ near zero). The FanGraphs "Great Valuation System Test" crowned SGP the winner by a wide margin, so the methodology isn't flawed — our implementation is.

This roadmap addresses four specific implementation gaps identified by comparing our system against the published methodology from Smart Fantasy Baseball, the FanGraphs test, and Todd Zola's SGP Theory:

1. **Mean-gap denominators** — our computation averages adjacent-team gaps, which is algebraically equivalent to `(1st - last) / (n-1)` and is dominated by outlier teams. The literature strongly favors regression-slope denominators.
2. **Ad-hoc volume weighting** — our rate-stat formula multiplies by `player_vol / avg_vol`, a linear approximation. The correct approach computes each player's marginal impact on a representative team's aggregate rate stat.
3. **No category weights infrastructure** — SGP lacks the configurable category weights that ZAR already supports, preventing experimentation with category emphasis without code changes.

Note on SV+HLD: the prior valuation-reform roadmap dropped SV+HLD (weight=0) based on its ρ = -0.608 correlation with WAR. However, this conflates "not correlated with WAR" with "not valuable in fantasy." SV+HLD is a real H2H category that teams need to win — saves and holds simply aren't captured by WAR. Dropping SV+HLD artificially inflates WAR ρ by making the system ignore a category WAR doesn't measure, but it also makes the system unable to value closers and high-leverage relievers, which is a real competitive disadvantage. **This roadmap keeps SV+HLD active** and instead fixes the underlying mechanisms that cause reliever overvaluation: the rate-stat formula (phase 2) and the budget split ([ZAR Pitcher Budget](zar-pitcher-budget.md) roadmap).

Each fix is independently testable. The final phase runs a full holdout comparison against baseline ZAR to determine if SGP can become the production system.

## Status

| Phase | Status |
|-------|--------|
| 1 — Regression-slope denominators | done (2026-03-11) |
| 2 — Team-impact rate-stat formula | done (2026-03-11) |
| 3 — Category weights and evaluation infrastructure | done (2026-03-11) |
| 4 — Holdout validation | done (2026-03-11) |

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

## Phase 3: Category weights and evaluation infrastructure

Add configurable category weights to SGP (matching ZAR's existing capability) and establish evaluation metrics that don't penalize role-based categories like SV+HLD.

### Context

SGP currently has no category weights infrastructure, unlike ZAR which supports an optional `category_weights` dict. Adding this to SGP enables experimentation — tuning category emphasis without code changes — even though the default should keep all categories at weight 1.0 (including SV+HLD).

The current evaluation framework gates primarily on WAR ρ, which is a poor target for validating SV+HLD-inclusive systems: WAR doesn't measure saves/holds, so any system that values relievers will look worse on WAR ρ by construction. Phase 4's validation needs supplementary metrics that capture reliever ranking quality without penalizing the system for valuing a real fantasy category.

### Steps

1. **Add category weights to `compute_sgp_scores()`**. Multiply `category_sgp[cat.key]` by `category_weights.get(cat.key, 1.0)` before summing composite. Reuse the same semantics as ZAR: `None` means all weights = 1.0. This enables experimentation but the default keeps all categories active.
2. **Wire category weights through `SgpModel`**. Read from `model_params.get("category_weights")` and pass to the pipeline, same as ZAR.
3. **Add a WAR-by-pool evaluation metric.** The current `war_correlation_pitchers` lumps SP and RP together. Add a `war_correlation_sp` metric (pitchers with ≥ some starts threshold) to isolate SP ranking quality from RP noise. This gives phase 4 a fairer target for systems that intentionally value RP via SV+HLD.
4. **Add a "category hit rate" metric.** For each pitching category, compute how well the top-N predicted contributors actually contributed. For SV+HLD specifically: do the top-20 projected SV+HLD contributors actually produce saves/holds? This measures whether the system values the *right* relievers, not whether it should value relievers at all.
5. **Add tests.** Verify category weights work in SGP. Verify weight=0 removes a category from composite. Verify new evaluation metrics produce sensible output.

### Acceptance criteria

- `compute_sgp_scores()` accepts optional `category_weights` with the same semantics as ZAR's implementation.
- Default behavior (no weights) produces identical output to current SGP (backward compatible).
- Evaluation framework includes SP-only WAR ρ and per-category hit-rate metrics.
- Phase 4 validation plan uses these supplementary metrics alongside overall WAR ρ.

---

## Phase 4: Holdout validation

Run the full comparison matrix: baseline SGP, SGP with regression denominators, SGP with team-impact rates, and the fully overhauled SGP — all against baseline ZAR.

### Context

Each prior phase produces an independently testable improvement. This phase measures their individual and combined effects on holdout accuracy and determines whether the overhauled SGP can become the production system.

The evaluation uses established independent targets (WAR ρ, top-N hit rates) via `fbm valuations compare --check`, supplemented by the SP-only WAR ρ and per-category hit-rate metrics from phase 3. All SGP configurations keep SV+HLD active — the comparison baseline is **baseline ZAR** (also SV+HLD active), not zar-reformed, to ensure an apples-to-apples comparison.

### Steps

1. **Define the test grid.** At minimum, evaluate these configurations on seasons 2024 and 2025:

   | Config | Denominators | Rate Stats | SV+HLD | min_ip |
   |--------|-------------|------------|--------|--------|
   | sgp-baseline | mean-gap | unweighted | active | 30 |
   | sgp-regression | regression | unweighted | active | 30 |
   | sgp-team-impact | mean-gap | team-impact | active | 30 |
   | sgp-overhauled | regression | team-impact | active | 30 |
   | zar (control) | n/a | n/a | active | 30 |

2. **Generate holdout valuations** for each configuration using ensemble projections.
3. **Run `fbm valuations compare`** for each candidate against baseline ZAR on both seasons. Record WAR ρ (overall, batter, pitcher), SP-only WAR ρ, top-25/50/100 hit rates, and per-category hit rates (especially SV+HLD).
4. **Build the comparison matrix.** Identify which fixes matter most. Expected rank order of impact: team-impact rates > regression denominators (the rate-stat formula is the dominant source of reliever inflation in SGP).
5. **Pitcher-focused inspection.** For `sgp-overhauled`: are the top-20 pitcher valuations plausible? Are SP aces valued appropriately? Are closers valued but not absurdly inflated (the $110 Edwin Díaz problem should be gone)?
6. **Reliever sanity check.** Top-10 RP valuations should be positive but modest — closers should appear in the draftable pool but not dominate it. Compare against ADP for reasonableness.
7. **Run regression gate checks** (`--check`) for `sgp-overhauled` vs baseline ZAR on both seasons.
8. **Adopt, blend, or reject:**
   - **Adopt** if `sgp-overhauled` matches or beats baseline ZAR on SP WAR ρ and overall hit rates, while producing more plausible reliever valuations.
   - **Blend** if SGP and ZAR have complementary strengths (e.g., SGP better on pitchers, ZAR better on batters) — consider an ensemble valuation.
   - **Reject** if `sgp-overhauled` still underperforms baseline ZAR, documenting which fixes helped and which didn't.
9. **Document results** in this roadmap's status table with the full comparison matrix.

### Acceptance criteria

- Comparison matrix covering all configurations on both holdout seasons with WAR ρ, SP WAR ρ, hit-rate, and per-category hit-rate metrics.
- Each phase's individual contribution is measured (regression denominators alone, team-impact rates alone).
- `sgp-overhauled` eliminates the reliever inflation problem: no RP valued above $40 (the $110 Díaz problem from baseline SGP is gone).
- SP WAR ρ for `sgp-overhauled` exceeds baseline SGP's pitcher WAR ρ on both seasons.
- SV+HLD category hit rate is at least as good as baseline ZAR (the system values the *right* relievers).
- Go/no-go decision documented with supporting data.

### Gate: go/no-go

**Go (adopt)** if `sgp-overhauled` matches or improves baseline ZAR's SP WAR ρ on both seasons without degrading batter metrics or hit rates, while producing plausible reliever valuations. **Go (blend)** if SGP and ZAR have complementary strengths — pursue an ensemble approach. **No-go** if `sgp-overhauled` still underperforms baseline ZAR on SP WAR ρ on both seasons despite all fixes. In that case, SGP's weakness may be fundamental to H2H category leagues (SGP was designed for and validated on roto, not H2H), and further investment should be redirected to [ZAR Pitcher Budget](zar-pitcher-budget.md) improvements instead.

### Results (original — superseded)

> **Note:** The original phase 4 results below were invalidated by a polluted projection pool bug discovered on 2026-03-12. See [Corrected results](#corrected-results-2026-03-12) below for the accurate holdout validation. The original results are preserved for reference.

All SGP configs used ensemble/routed-sgbm projections to match the ZAR baseline population (~945 players per season). Control: `zar/holdout-baseline`.

#### Comparison matrix (original — polluted pool)

| Config | Season | WAR ρ (all) | WAR ρ (SP) | Hit-25 | Hit-50 | Hit-100 | Regression gate |
|--------|--------|-------------|------------|--------|--------|---------|-----------------|
| zar (control) | 2024 | 0.427 | 0.414 | 40% | 36% | 51% | — |
| zar (control) | 2025 | 0.480 | 0.476 | 48% | 44% | 47% | — |
| sgp-baseline | 2024 | 0.435 (+0.008) | 0.436 (+0.021) | 36% | 36% | 42% | — |
| sgp-baseline | 2025 | 0.461 (-0.019) | 0.438 (-0.038) | 40% | 42% | 42% | — |
| sgp-regression | 2024 | 0.435 (+0.008) | 0.427 (+0.013) | 32% | 38% | 42% | — |
| sgp-regression | 2025 | 0.467 (-0.014) | 0.433 (-0.043) | 40% | 44% | 42% | — |
| sgp-team-impact | 2024 | 0.434 (+0.006) | 0.416 (+0.002) | 40% | 38% | 42% | — |
| sgp-team-impact | 2025 | 0.460 (-0.021) | 0.411 (-0.065) | 40% | 44% | 43% | — |
| sgp-overhauled | 2024 | 0.433 (+0.006) | 0.411 (-0.004) | 40% | 42% | 43% | PASS |
| sgp-overhauled | 2025 | 0.467 (-0.013) | 0.423 (-0.053) | 40% | 46% | 43% | FAIL |

Per-category hit rates (sgp-overhauled vs ZAR): HR, R, RBI, SO, W all matched ZAR. OBP consistently -10 to -15pp worse. SB -5pp worse. ERA and WHIP 0% for both systems (expected — these rate stats have noisy actuals). SV+HLD not evaluable (combined stat not tracked in actuals).

#### Polluted pool bug (discovered 2026-03-12)

The original sgp-overhauled holdout runs were generated via a script that fetched projections using `get_by_season(season, system="ensemble")` instead of `get_by_system_version("ensemble", "routed-sgbm")`. The former returns *all* ensemble versions for that season, so the model received projections from both `routed-sgbm` (2269 pitchers) and `routed-sgbm-pt` (1615 pitchers) — many of the same players counted twice with different projection versions.

This inflated the pitcher pool from ~600 (correct) to ~1450, which:

1. **Raised the replacement level** — more players competing for 96 roster spots (8 pitcher slots × 12 teams) pushes the "best freely available" baseline higher.
2. **Compressed top-end dollar values** — Skenes went from $162 (correct pool) to $72 (polluted pool) because the surplus above replacement was smaller with a higher replacement level.
3. **Corrupted the OBP team-impact fallback** — the projection-based `representative_team` OBP entry had doubled total PA, inflating the team volume used in the marginal-impact formula.

The compressed values made SGP look more reasonable in pitcher inspection ("no RP above $40", "plausible reliever valuations") but this was an artifact — the real SGP values with a correct single-version pool are much more top-heavy.

### Corrected results (2026-03-12)

After fixing the polluted pool, sgp-overhauled was regenerated using `get_by_system_version("ensemble", "routed-sgbm")` for both holdout seasons, producing ~945 players per season (matching ZAR's population).

#### Corrected comparison matrix

| Config | Season | Value MAE | Rank ρ | WAR ρ (all) | WAR ρ (batters) | WAR ρ (pitchers) | WAR ρ (SP) | Hit-25 | Hit-50 | Hit-100 |
|--------|--------|-----------|--------|-------------|-----------------|------------------|------------|--------|--------|---------|
| zar (control) | 2024 | 3.83 | 0.685 | 0.426 | 0.450 | 0.391 | 0.414 | 40% | 36% | 52% |
| sgp-overhauled | 2024 | 4.15 | 0.689 | 0.448 | 0.459 | 0.421 | 0.436 | 36% | 40% | 45% |
| zar (control) | 2025 | 4.11 | 0.680 | 0.479 | 0.510 | 0.425 | 0.476 | 48% | 44% | 46% |
| sgp-overhauled | 2025 | 4.37 | 0.671 | 0.476 | 0.495 | 0.435 | 0.446 | 40% | 40% | 46% |

Shared population (min PA ≥ 50): n=875 (2024), n=762 (2025).

Per-category hit rates were identical between systems for all categories on both seasons (HR, R, RBI, OBP, SB, ERA, SO, SV+HLD, W, WHIP). The OBP gap from the original results (-10 to -15pp) was entirely a population artifact — with shared populations, both systems produce the same OBP hit rate.

#### Corrected analysis

With a clean pool, the picture changes:

- **2024:** SGP wins WAR ρ overall (+0.022), batters (+0.009), pitchers (+0.030), and SP (+0.022). ZAR wins Value MAE (-0.32) and hit rate top-100 (-7pp). Mixed result.
- **2025:** ZAR wins most metrics — Value MAE (-0.26), Rank ρ (+0.009), WAR ρ overall (+0.003), batters (+0.015), SP (+0.030), hit rates top-25 (-8pp) and top-50 (-4pp). SGP only wins pitcher WAR ρ (+0.010).

#### Structural pitcher overvaluation

With the correct pool, SGP's dollar values reveal a structural problem:

| Version | Season | Top pitcher | Top batter | Max pitcher / Max batter |
|---------|--------|-------------|------------|--------------------------|
| sgp-overhauled | 2024 | $175.0 | $89.4 | 1.96× |
| sgp-overhauled | 2025 | $160.5 | $82.2 | 1.95× |
| sgp/production | 2026 | $162.7 | $73.2 | 2.22× |
| zar/production | 2026 | $73.3 | $66.8 | 1.10× |

SGP's top pitcher is consistently ~2× the top batter, while ZAR's are roughly equal. This is a known structural limitation of SGP documented in the fantasy baseball analytics community (Smart Fantasy Baseball, FanGraphs, Pitcher List).

**Root cause:** Elite starting pitchers accumulate SGP in *both* counting categories (SO, W) and rate categories (ERA, WHIP) simultaneously. A pitcher like Skenes gets ERA SGP of 6.87 (team-impact: adding 194 IP of 2.92 ERA to a 1033 IP team at 3.70 ERA), WHIP SGP of 1.30, SO SGP of 7.36, and W SGP of 4.51, for a composite of 20.03. No batter can accumulate equivalent composite because batter rate stats (OBP) don't compound with counting stats the same way.

This is not a bug in the rate-stat formula — the team-impact calculation is correct. It's an inherent property of how SGP combines rate and counting dimensions for pitchers. ZAR avoids this because its z-score approach normalizes each category independently and produces a more compressed distribution.

**Attempted mitigation — linearized rate stats:** Converting ERA to "earned runs saved" (`IP × avg_ERA/9 - ER`) and WHIP to "baserunners prevented" (`IP × avg_WHIP - (BB+H)`) eliminates the team-impact nonlinearity but produces a similar spread (#1 to #96 spread of 11.58 SGP vs 9.84 for team-impact). The top pitcher still reaches $129 — the concentration is driven by elite pitchers dominating both dimensions, not by the specific rate-stat formula.

#### Go/no-go decision: **No-go** (confirmed)

The corrected results confirm the no-go decision, though for partially different reasons than originally documented:

1. **Structural pitcher overvaluation** — SGP produces top pitchers at $160-175 (2× top batters). This makes the dollar values unsuitable for auction guidance without post-hoc rescaling.
2. **Mixed holdout accuracy** — SGP is competitive with ZAR on WAR ρ (wins 2024, loses 2025) but consistently worse on Value MAE and top-N hit rates. No clear accuracy advantage to justify the dollar-scale distortion.
3. **Category hit rates are identical** — the original OBP gap was a population artifact from the polluted pool and the pre-fix evaluation infrastructure (independent populations per system). With shared-population comparison, all category hit rates match.

**Recommendation:** Keep SGP infrastructure for research and as a supplementary ranking signal. The WAR ρ results suggest SGP captures pitcher value slightly differently than ZAR (better pitcher WAR ρ on both seasons), which could be useful in an ensemble. However, SGP dollar values should not be used directly for auction budgets due to the structural pitcher concentration.

### Evaluation infrastructure improvements (2026-03-12)

Three improvements to the valuation comparison framework were implemented during this investigation:

1. **Shared-population comparison** — `compare()` now evaluates both systems on the intersection of players present in both predictions AND in actuals. Previously, each system was evaluated independently with different matched sets, making metrics asymmetric and incomparable. This eliminated the OBP hit-rate artifact (15pp gap that was entirely population-dependent).

2. **Min PA/IP filtering** — `evaluate()` and `compare()` accept optional `min_pa` and `min_ip` parameters to filter actual batter/pitcher pools. Low-PA noise players with extreme rate stats (e.g., .667 OBP from 6 PA) distorted rate-stat category evaluations. CLI flags: `--min-pa`, `--min-ip`.

3. **Derived stats in actuals (SV+HLD)** — `add_derived_stats()` computes compound stats (SV+HLD = SV + HLD) on actual stat dicts, making the combined category evaluable in hit-rate metrics. Extracted from `projection_lookup.py` to `stats_conversion.py` for shared use.

---

## Ordering

- **Phases 1, 2, and 3** are independent and can be implemented in any order or in parallel. Phase 1 changes denominator computation, phase 2 changes rate-stat scoring, phase 3 adds category weights and evaluation metrics. None modify shared code paths.
- **Phase 4** depends on all prior phases (needs all configurations available for the grid search).
- This roadmap is independent of [ZAR Pitcher Budget](zar-pitcher-budget.md) — they address the same symptom (pitcher overvaluation) from different angles (SGP as an alternative system vs ZAR parameter tuning). If both produce improvements, the better system wins; if both improve different aspects, an ensemble is possible.
- The [SGP Rate-Stat Volume Weighting](sgp-rate-stat-volume-weighting.md) roadmap's phase 2 (deferred holdout validation) is superseded by this roadmap's phase 4, which tests a strictly better rate-stat approach (team-impact formula subsumes volume weighting).
- All SGP configurations in this roadmap keep SV+HLD active. The comparison baseline is baseline ZAR (not zar-reformed) to ensure an apples-to-apples comparison where both systems value all league categories.
