# SGP Rate-Stat Volume Weighting Roadmap

SGP's rate-stat handling treats all pitchers equally regardless of innings pitched: a reliever with 60 IP and 2.80 ERA gets the same ERA SGP score as a starter with 180 IP and 2.80 ERA. This is because `compute_sgp_scores` divides `(baseline - player_rate)` by the SGP denominator without any volume factor, unlike ZAR which multiplies by IP to produce marginal contributions.

The investigation notebook (`notebooks/sgp_denominator_investigation.ipynb`) confirmed the impact: SGP's top-valued pitcher is a reliever at $110 (vs $17 in ZAR), pitcher WAR ρ is 0.377 (vs ZAR's 0.458), and rate stats account for 62% of RP composite scores despite RPs pitching only 65 IP on average. A simple IP-weighting simulation improved SGP valued-pitcher WAR ρ from 0.377 to 0.572, surpassing ZAR. This roadmap implements and validates that fix.

This is orthogonal to the [Optimal Position Assignment](optimal-position-assignment.md) roadmap: that fixes *how many* pitchers get valued (pool size), while this fixes *which* pitchers rank highest (rate-stat distortion). Both are needed for SGP to be competitive with ZAR on pitcher accuracy.

## Status

| Phase | Status |
|-------|--------|
| 1 — Volume-weighted rate-stat SGP scores | done (2025-07-11) |
| 2 — Holdout validation and adoption | not started |

## Phase 1: Volume-weighted rate-stat SGP scores

Modify `compute_sgp_scores` to weight rate-stat contributions by the player's volume (IP for pitching, PA for batting) relative to the pool average.

### Context

The current SGP rate-stat formula in `engine.py` (lines 81–94):

```python
# Lower-is-better (ERA, WHIP):
sgp = (baseline - player_rate) / abs(denom)
# Higher-is-better (OBP):
sgp = (player_rate - baseline) / denom
```

The baseline is `statistics.median(rates)` — also unweighted. Neither step accounts for how much a player contributes to a team's aggregate rate stat. A team's ERA is IP-weighted across its pitchers, so a starter with 180 IP has 3x the influence on team ERA as a reliever with 60 IP. The fix multiplies each player's rate-stat SGP by `player_volume / pool_avg_volume`:

```python
# Proposed:
volume_weight = stats.get(cat.denominator, 0.0) / avg_volume
sgp = (baseline - player_rate) / abs(denom) * volume_weight
```

This preserves the SGP framework (scores still represent standings-gain units) while correctly scaling each player's contribution by their share of the team's innings or plate appearances.

The baseline should also switch from unweighted median to volume-weighted mean (matching ZAR's approach), since the pool's aggregate rate is inherently volume-weighted.

### Steps

1. **Add a `volume_weighted` parameter to `compute_sgp_scores`.** Default to `False` for backward compatibility. When `True`:
   - Compute baseline as volume-weighted mean instead of unweighted median (use `cat.denominator` to look up each player's volume from `stats_list`)
   - After computing each player's raw rate-stat SGP, multiply by `player_volume / avg_volume` where `avg_volume` is the mean volume across all players in the pool
   - Counting stats are unaffected — volume weighting only applies to `StatType.RATE` categories

2. **Update `run_sgp_pipeline` to pass through the `volume_weighted` flag.**

3. **Update `SgpModel` to accept `volume_weighted` as a model parameter.** Read from `model_params.get("volume_weighted", False)` and pass to the pipeline. This lets the holdout comparison run both variants.

4. **Update existing tests.** The test `test_rate_stat_independent_of_ip` validates the current unweighted behavior — it should continue to pass with `volume_weighted=False`. Add a parallel test `test_rate_stat_scales_with_ip` that verifies volume weighting: two pitchers with identical ERA but different IP should get different SGP scores proportional to their IP ratio.

5. **Add edge case tests:**
   - Player with zero volume (IP=0) gets SGP=0 for rate stats (already handled by the `denom_val <= 0` guard)
   - Single player in pool (avg_volume = their volume, weight = 1.0)
   - Volume-weighted baseline differs from median baseline for skewed pools (many low-IP relievers pull the unweighted median toward RP-like rates)
   - Batters: OBP volume weighting uses PA as the volume field

### Acceptance criteria

- `compute_sgp_scores(volume_weighted=False)` produces identical results to the current implementation (no regression).
- `compute_sgp_scores(volume_weighted=True)` scales rate-stat SGP by `player_volume / avg_volume`.
- Two pitchers with identical ERA but IP of 180 and 60 get rate-stat SGP scores in approximately 3:1 ratio (with `volume_weighted=True`).
- Baseline for rate stats is volume-weighted mean when `volume_weighted=True`.
- Counting stat scores are unaffected by the flag.
- All existing SGP tests pass without modification.

---

## Phase 2: Holdout validation and adoption

Validate the volume-weighted SGP variant against holdout seasons using the existing evaluation framework, compare to ZAR-reformed, and adopt if improved.

### Context

The investigation notebook showed IP-weighted SGP improving valued-pitcher WAR ρ from 0.377 to 0.572 in a quick simulation. However, that simulation was rough — it applied post-hoc weighting to existing SGP scores rather than running the full pipeline with volume-weighted baselines and proper replacement/VAR/dollar conversion. This phase runs the real pipeline and validates with the established protocol.

The evaluation framework (`fbm valuations compare`) already supports comparing any two valuation systems on independent targets: WAR ρ (overall, batter, pitcher), top-N hit rates (25, 50, 100), and pitcher-focused metrics.

### Steps

1. **Generate holdout valuations** for volume-weighted SGP on both holdout seasons:
   ```bash
   fbm predict sgp --season 2024 --param league=h2h --param projection_system=steamer --param volume_weighted=true --version holdout-volume-weighted
   fbm predict sgp --season 2025 --param league=h2h --param projection_system=steamer --param volume_weighted=true --version holdout-volume-weighted
   ```

2. **Run evaluation framework comparisons** against both the current SGP and ZAR-reformed baselines:
   ```bash
   fbm valuations compare sgp/holdout sgp/holdout-volume-weighted --season 2024 --league h2h --check
   fbm valuations compare sgp/holdout sgp/holdout-volume-weighted --season 2025 --league h2h --check
   fbm valuations compare zar-reformed/holdout sgp/holdout-volume-weighted --season 2024 --league h2h --check
   fbm valuations compare zar-reformed/holdout sgp/holdout-volume-weighted --season 2025 --league h2h --check
   ```

3. **Pitcher-focused analysis.** For each system, examine:
   - Top-20 pitcher valuations: are starters now properly valued above similarly-rated relievers?
   - SP vs RP dollar distribution: does the balance shift toward starters?
   - Rate-stat SGP contribution: has the RP rate-stat inflation been eliminated?

4. **Batter impact check.** Volume weighting also affects OBP (via PA). Verify batter rankings and WAR ρ are not degraded — OBP volume weighting should be beneficial (high-PA batters get more OBP credit) but needs confirmation.

5. **Adopt if improved.** If volume-weighted SGP matches or improves all independent targets on both holdout seasons:
   - Make `volume_weighted=True` the default for SGP
   - Regenerate 2026 production SGP valuations
   - Update the [Optimal Position Assignment](optimal-position-assignment.md) roadmap's phase 4 to include the volume-weighted SGP variant in its comparison matrix

6. **Document results** in this roadmap's status table with go/no-go decision and key metrics.

### Acceptance criteria

- Comparison matrix covering volume-weighted SGP vs original SGP vs ZAR-reformed on both 2024 and 2025 holdout seasons.
- Pitcher WAR ρ improves over original SGP on both seasons (expected: significant improvement based on investigation).
- Pitcher WAR ρ is competitive with ZAR-reformed (>= 0.218 on 2024, >= 0.248 on 2025 — the ZAR-reformed baselines).
- Top-N hit rates do not regress on any N value vs original SGP.
- Batter WAR ρ does not regress vs original SGP.
- If adopted, `volume_weighted=True` becomes the default and 2026 valuations are regenerated.

### Gate: go/no-go

**Go** if volume-weighted SGP improves pitcher WAR ρ on both holdout seasons without degrading batter accuracy or hit rates. **No-go** if pitcher WAR ρ regresses on either season — this would indicate the volume weighting overcorrects (unlikely given the investigation results, but must be verified through the full pipeline). If no-go, document findings for potential alternative approaches (marginal-contribution SGP or separate SP/RP pools).

---

## Ordering

- **Phase 1** (implementation) has no dependencies and can start immediately.
- **Phase 2** (validation) depends on phase 1 and requires holdout projection data (already available from the valuation-reform roadmap).
- This roadmap is **independent of** the [Optimal Position Assignment](optimal-position-assignment.md) roadmap. Both fix different aspects of SGP and can be implemented in any order. When both are complete, their combined effect on SGP should be validated together (optimal assignment's phase 4 already includes SGP reassessment).
- The valuation-reform roadmap's SGP settings (category selection, min_ip = 60) remain in effect — this roadmap only changes rate-stat scoring, not category signals.
