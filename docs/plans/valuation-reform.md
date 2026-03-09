# Valuation Reform Roadmap

The pitcher valuation diagnosis (`notebooks/pitcher_valuation_diagnosis.ipynb`, 2026-03-09) revealed that ZAR's ranking problems are structural, not fixable by layering adjustments (injury discount, breakout/bust) on top of the existing formula. Five interacting issues were identified:

1. **SV+HLD anti-predicts real value.** SV+HLD z-score has ρ = −0.608 against actual WAR. Removing it improves pitcher WAR correlation from 0.24 to 0.43 — the single largest improvement of any variant tested.
2. **SO and W are pure IP proxies.** SO correlates r = 0.954 with IP, W correlates r = 0.984. They add no ranking information beyond workload.
3. **Rate stat marginal conversion is broken.** `(baseline_rate − player_rate) × IP` makes ERA/WHIP z-scores volume-dependent, conflating skill with workload.
4. **Pool composition skew.** 586 pitchers in the pool, 68% with IP < 80 (relievers). Starters get extreme z-scores purely from volume.
5. **min_ip = 30 is too low.** Raising to 60 consistently improves accuracy.

The circular evaluation target problem (actual ZAR$ uses the same flawed formula as predicted ZAR$) means we cannot validate formula changes against actual ZAR$. This roadmap builds SGP-based valuation as a principled alternative, applies targeted ZAR formula fixes, and validates both against independent targets.

## Status

| Phase | Status |
|-------|--------|
| 1 — SGP denominator computation | not started |
| 2 — SGP valuation engine | not started |
| 3 — ZAR category signal reform | not started |
| 4 — Head-to-head validation and production adoption | not started |

## Phase 1: SGP denominator computation

Compute per-category SGP (Standings Gain Points) denominators from real league standings data.

### Context

SGP denominators answer: "how many standings points does one additional unit of a category buy?" For example, if the average gap between adjacent teams in HR is 25, then SGP(HR) = 25 — each additional 25 HR is worth roughly one standings point. This replaces the z-score approach (which measures distance from the population mean in stdev units) with a measure calibrated to actual league outcomes.

The [League Standings Import](league-standings-import.md) roadmap provides the raw data: per-team season category totals for 4–5 seasons of the redraft league. This phase transforms that raw data into SGP denominators.

### Steps

1. **Design the SGP computation.** For each category and season:
   - Sort the 12 teams by their category total
   - Compute the difference between adjacent teams (11 gaps)
   - Take the mean of those gaps as the season's SGP denominator for that category
   - For rate stats (ERA, WHIP, OBP), compute on the raw rate (not IP-weighted marginals) — SGP naturally handles rate/counting distinction
2. **Average across seasons.** Use the mean SGP denominator across all available seasons (4–5 years) for stability. Optionally weight recent seasons higher.
3. **Create an SGP config module.** Store computed denominators in a format the valuation engine can consume — either as a config section in `fbm.toml` or as a computed artifact. Include the raw per-season values for inspection.
4. **Handle rate stat direction.** ERA and WHIP are "lower is better" — SGP denominators should be positive but the valuation math must invert direction. Document the convention clearly.
5. **Add a CLI command or report.** `fbm sgp denominators --league redraft` — displays the computed denominators per category with per-season breakdowns.
6. **Add tests.** Verify SGP computation on synthetic standings data with known gaps.

### Acceptance criteria

- SGP denominators computed for all 10 categories (HR, R, RBI, OBP, SB, ERA, WHIP, SO, W, SV+HLD).
- Per-season and cross-season averages are available.
- Rate stats (ERA, WHIP, OBP) produce sensible denominators (not distorted by IP/PA weighting).
- Denominators are consumable by the valuation engine in phase 2.

### Dependencies

- [League Standings Import](league-standings-import.md) phase 2 (historical standings data must be imported first).

---

## Phase 2: SGP valuation engine

Build a new valuation system (`sgp`) that converts projections to auction dollars using SGP denominators instead of z-scores.

### Context

SGP valuation works differently from ZAR:
- **ZAR**: compute z-score per category (distance from mean in stdev units), sum z-scores, subtract replacement level, convert to dollars.
- **SGP**: compute marginal standings points per category (player's projected stats / SGP denominator), sum marginal standings points, subtract replacement level, convert to dollars.

The key advantage: SGP denominators are calibrated to real league outcomes, so 1 HR of value ≈ 1 SB of value ≈ 1 W of value in terms of standings impact. ZAR's z-score approach can dramatically over/underweight categories when the population distribution doesn't match the standings distribution (which is exactly the SV+HLD problem — high variance in the population creates large z-scores, but the standings impact is modest).

For rate stats, SGP handles them naturally: a team's ERA of 3.50 vs 3.75 is a fixed standings gap regardless of IP. This eliminates the broken marginal conversion (`(baseline − player) × IP`) that makes ZAR rate stats volume-dependent.

### Steps

1. **Design the SGP model.** Create `SgpModel` in `models/sgp/` following the same pattern as `ZarModel`:
   - Accept league config and SGP denominators as inputs
   - For counting stats: `marginal_sgp = projected_stat / sgp_denominator`
   - For rate stats: `marginal_sgp = (league_avg_rate − player_rate) / sgp_denominator` (sign-corrected for direction)
   - Sum marginal SGP across categories → composite SGP
   - Apply replacement level and dollar conversion (same math as ZAR's VAR→dollars step)
2. **Handle the pitcher flex slot (P).** The league has 2 SP + 2 RP + 4 P slots. SGP must rank all pitchers in a unified pool (same as ZAR today) since P slots accept either SP or RP. Use the same roster-slot counting as ZAR.
3. **Register as a first-class model.** `fbm predict sgp --season 2026 --param league=h2h --version production` should produce and persist valuations.
4. **Sanity checks.** Verify: total dollars sum to budget ($260 × 12), dollar distribution has a reasonable shape (top players $30–40, not $100+), batters/pitchers split is reasonable.
5. **Add tests.** Unit tests with synthetic projections and known SGP denominators. Integration test verifying dollar sum equals budget.

### Acceptance criteria

- `fbm predict sgp` produces auction dollar valuations using SGP denominators.
- Dollar values sum to the league budget (within $1 tolerance due to rounding).
- Rate stats are not IP/PA-weighted — a pitcher's ERA value depends only on their ERA, not their IP.
- Top player values are in a reasonable range ($25–45).

---

## Phase 3: ZAR category signal reform

Apply the diagnostic findings as targeted fixes to the ZAR formula, creating a `zar-reformed` variant.

### Context

Even if SGP becomes the primary valuation system, ZAR reform is valuable for two reasons: (1) ZAR is the existing production system and incremental improvement is less risky than a wholesale replacement, and (2) comparing reformed ZAR against SGP tells us whether the problems are in the z-score methodology itself or just in the current parameterization.

The prototyped variants showed that removing SV+HLD and raising min_ip are the two highest-impact changes. Category weighting and counting stat deduplication had smaller effects but are worth including.

### Steps

1. **Add configurable category weights to ZAR.** Currently, composite z = unweighted sum of per-category z-scores. Add an optional `weights` dict to the ZAR pipeline that multiplies each category's z-score before summing. Default weights = 1.0 (backward compatible). Configure via `fbm.toml` league section.
2. **Downweight SV+HLD.** Set default weight for SV+HLD to 0.0 (effectively removing it). The diagnosis showed this is the single most impactful change (+0.19 WAR ρ improvement). Make it configurable so leagues that value saves differently can adjust.
3. **Raise default min_ip to 60.** The diagnosis showed min_ip=60 consistently outperforms min_ip=30 across all metrics. Update the league config default. This reduces the pitcher pool from ~586 to ~350, removing the low-IP relievers that skew z-scores.
4. **Create `zar-reformed` model variant.** Register as a first-class model that uses `ZarModel` with the reformed defaults (SV+HLD weight=0, min_ip=60). This preserves the baseline `zar` for comparison.
5. **Add tests.** Verify category weights are applied correctly. Verify min_ip=60 excludes the expected pitchers. Verify backward compatibility — `zar` with no weight config produces identical output to today.

### Acceptance criteria

- `fbm predict zar-reformed` produces valuations with SV+HLD removed and min_ip=60.
- `fbm predict zar` (baseline) produces identical output to current production (backward compatible).
- Category weights are configurable in `fbm.toml` per league.
- Tests confirm weight=0 for a category produces the same result as omitting that category entirely.

### Dependencies

- [Evaluation Framework](evaluation-framework.md) phase 2 (independent targets) should be in place before validation, but the implementation itself has no hard dependencies.

---

## Phase 4: Head-to-head validation and production adoption

Compare baseline ZAR, reformed ZAR, and SGP on holdout seasons using independent evaluation targets. Adopt the winning system.

### Context

The [Evaluation Framework](evaluation-framework.md) roadmap provides the tools: fantasy-relevant filtering (phase 1), independent targets like WAR correlation and top-N hit rate (phase 2), batter/pitcher stratification (phase 3), and a compare command (phase 4). This phase uses those tools to make a data-driven decision.

The critical insight from the diagnosis: actual ZAR$ is a circular target, so improvements must be measured against WAR, SGP composite, and top-N hit rate. A system that improves WAR ρ from 0.03 to 0.15 but shows no change on actual ZAR$ is a genuine improvement that the old evaluation would miss.

### Steps

1. **Generate holdout valuations for all systems:**
   ```bash
   for sys in zar zar-reformed sgp; do
     fbm predict $sys --season 2024 --param league=h2h --param projection_system=steamer --version holdout
     fbm predict $sys --season 2025 --param league=h2h --param projection_system=steamer --version holdout
   done
   ```
2. **Evaluate using independent targets.** Use the evaluation framework's compare command (or manual evaluation if the compare command isn't ready):
   - WAR ρ (overall, batters, pitchers)
   - Top-N hit rate (N = 25, 50, 100)
   - Fantasy-relevant MAE and ρ (acknowledging this is circular for ZAR but not for SGP)
3. **Build a comparison matrix.** System × season × metric. Highlight statistically significant differences.
4. **Specific pitcher focus.** For each system, examine the top 20 pitcher overvaluations and undervaluations. Does the system correctly value high-WAR pitchers? Does it avoid inflating SV+HLD-driven relievers?
5. **Adopt the winner.** If SGP or zar-reformed clearly outperforms baseline ZAR on independent targets across both holdout seasons, adopt it as the production default for 2026.
6. **Document results** in this roadmap's status section, following the same format as the valuation-accuracy roadmap phase 2 results.

### Acceptance criteria

- Comparison matrix covering all systems on both holdout seasons with at least 3 independent targets.
- Pitcher WAR ρ improves over baseline ZAR (from ~0.03 to >0.10 on fantasy-relevant players).
- Top-N hit rate improves for at least one N value.
- The chosen production system is adopted and 2026 valuations regenerated.

### Gate: go/no-go

**Go** if either SGP or zar-reformed improves pitcher WAR ρ by >0.05 and top-50 hit rate by >5% on at least one holdout season without degrading batter metrics. **No-go** if neither system improves independent targets, indicating the problem is in the projections rather than the valuation formula. If no-go, document findings and redirect effort to projection accuracy.

## Ordering

- **Phase 1** depends on [League Standings Import](league-standings-import.md) (needs real standings data).
- **Phase 2** depends on phase 1 (needs SGP denominators).
- **Phase 3** has no hard dependencies and can be implemented in parallel with phases 1–2.
- **Phase 4** depends on phases 2–3 (needs both systems to compare) and benefits from [Evaluation Framework](evaluation-framework.md) phases 1–2 (filtering + independent targets), though manual evaluation is possible without them.
- The [Valuation Accuracy](valuation-accuracy.md) roadmap phases 3–4 (breakout/bust integration, combined validation) can stack on top of whichever system wins here.
