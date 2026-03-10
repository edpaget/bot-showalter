# Pitcher IP Distribution Roadmap

Pitcher valuations suffer from unrealistic IP distributions in the projection inputs. Steamer projects 1624 pitchers for 2024 with an average of 32.5 IP, but only 78 reach 150 IP and 1164 are under 50 IP. Actual seasons look very different: 725 pitchers averaging 54.4 IP, with 70 reaching 150 IP and only 393 under 50 IP. The heavy tail of low-IP pitchers distorts ZAR baselines and inflates rate stat contributions from relievers with tiny workloads.

The playing-time model already exists and predicts IP via OLS regression on historical IP, WAR, age curves, and external projection systems (steamer + ZiPS consensus). It produces a moderately better distribution (avg 44.5 IP, 175 pitchers at 100+ IP) but still overprojects the number of pitchers who will actually appear. This roadmap makes the playing-time model's IP predictions usable as the IP source in routed ensemble valuations, then tunes the model to produce a distribution that more closely matches actual seasons — so that pitcher valuations reflect realistic workloads without needing artificial pool caps like `min_ip`.

## Status

| Phase | Status |
|-------|--------|
| 1 — Baseline comparison and target calibration | in progress |
| 2 — PT-model IP in routed ensemble valuations | not started |
| 3 — Distribution-aware IP calibration | not started |
| 4 — Validation on holdout seasons | not started |

## Phase 1: Baseline comparison and target calibration

Produce a quantitative picture of how steamer, the playing-time model, and actual seasons differ in their pitcher IP distributions, and define concrete distributional targets for the playing-time model to hit.

### Context

We know the distributions differ qualitatively (steamer overprojects the number of pitchers, the PT model is closer but still too broad), but we don't have precise targets. Before tuning anything, we need to know what "good" looks like across multiple seasons, and whether the PT model's existing predictions are already close enough at certain thresholds.

### Steps

1. Create an analysis notebook (`notebooks/pitcher_ip_distribution_baseline.ipynb`) comparing IP distributions across steamer, playing-time model, and actuals for seasons 2019-2024.
2. For each source and season, compute: total pitchers with IP > 0, IP > 30, IP > 60, IP > 100, IP > 150; mean/median/std of IP; Gini coefficient or similar concentration metric; total pool IP.
3. Compute the IP distribution at key percentiles (p10, p25, p50, p75, p90) for each source to characterize the shape.
4. Overlay CDFs for all three sources on the same plot per season to visualize divergence.
5. Define target ranges for the PT model's output distribution based on the average actual-season shape across 2019-2024. These become the acceptance criteria for phase 3.

### Acceptance criteria

- Notebook with side-by-side distributional metrics for steamer, PT model, and actuals across 6 seasons.
- CDF overlay plots for each season.
- Written target ranges for key distribution metrics (count at IP thresholds, mean/median IP, total pool IP) derived from actuals.

## Phase 2: PT-model IP in routed ensemble valuations

Wire the playing-time model's IP predictions into the routed ensemble so that valuations can use PT-model IP instead of steamer IP, while keeping rate stats from statcast-gbm-preseason and counting stats derived from the PT-adjusted IP.

### Context

The ensemble's routed mode currently routes counting stats (including IP) to steamer and rate stats to statcast-gbm-preseason. The ensemble already supports a `playing_time` parameter that can override IP via `ConsensusLookup` / `normalize_projection_pt()`, but in routed mode the consensus PT override is not applied — the `routed()` engine function just picks raw stats from the named system. We need to apply PT normalization *after* routing so that IP comes from the PT model while counting stats scale proportionally.

### Steps

1. Add a post-routing PT normalization step in `EnsembleModel.predict()`: after `routed()` produces the merged stat dict, if consensus PT is available for a pitcher, rescale counting stats by `pt_model_ip / routed_ip` and set IP to the PT model's value. Rate stats remain untouched (they come from statcast-gbm-preseason via `use_direct_rates`).
2. Add tests verifying that routed + PT override produces: (a) IP from PT model, (b) counting stats scaled proportionally, (c) rate stats unchanged.
3. Add a `--playing-time` CLI parameter passthrough for the ensemble predict command if not already present (it may already exist from the playing-time-flexibility work).
4. Generate test valuations: `fbm predict ensemble --version routed-sgbm-pt --param playing_time=playing_time --param league=h2h` for 2024 and verify the output IP distribution matches the PT model's predictions.

### Acceptance criteria

- Ensemble routed mode with `playing_time=playing_time` produces projections where pitcher IP comes from the PT model.
- Counting stats are scaled proportionally to the IP change; rate stats are unchanged.
- CLI command works end-to-end: `fbm predict ensemble --version routed-sgbm-pt --param playing_time=playing_time`.
- Unit tests cover the post-routing normalization path.

## Phase 3: Distribution-aware IP calibration

Tune the playing-time model to produce a pitcher IP distribution that more closely matches actual seasons, using the targets defined in phase 1.

### Context

The PT model currently predicts IP via OLS regression with features including historical IP lags, WAR, age curves, starter ratio, and consensus IP from steamer+ZiPS. Its output distribution is closer to reality than raw steamer but still includes ~1600 pitchers when actual seasons have ~700. The model needs calibration at two levels: (a) who gets meaningful IP (a participation/threshold question), and (b) how much IP those pitchers get (a volume question).

### Steps

1. **Analyze PT model residuals** by IP tier: are the errors uniform, or does the model systematically overproject IP for low-IP pitchers and underproject for high-IP starters? This determines whether we need a global shift or tier-specific adjustments.
2. **Experiment with participation thresholds**: test whether adding features that predict "will this pitcher get any MLB innings" (e.g., roster status, 40-man status, prospect ranking, prior-season appearance count) improves the distribution shape. Use the experiment skill for playing-time to test candidates.
3. **Experiment with distributional post-processing**: test quantile-mapping or isotonic regression to map the PT model's raw IP predictions onto the actual-season IP CDF. This preserves the model's ranking while fixing the distribution shape.
4. **Evaluate calibration**: compare the tuned model's IP distribution against the phase 1 targets on holdout seasons.
5. Select the best approach (feature-based vs. post-processing vs. both) based on holdout performance.

### Acceptance criteria

- Tuned PT model's IP distribution hits the target ranges from phase 1 on at least 2 holdout seasons.
- Count of pitchers with IP > 0 is within 20% of the actual-season count.
- Mean and median IP are within 10% of actual-season values.
- The IP ranking (which pitchers get the most IP) does not regress: rank correlation with actuals must not decrease vs. the untuned model.

## Phase 4: Validation on holdout seasons

Generate end-to-end valuations using the calibrated PT-model IP distribution with direct rate stats, and compare against the current best (steamer IP + direct rates) and the steamer-only baseline.

### Context

This is the payoff phase: does a realistic IP distribution combined with statcast-gbm rate stats produce better valuations? The hypothesis is that pitcher valuations improve because (a) the pool composition is more realistic (fewer phantom low-IP pitchers dragging baselines), and (b) high-IP starters get appropriately large rate stat contributions.

### Steps

1. Generate routed ensemble projections with PT-model IP for 2024 and 2025: `fbm predict ensemble --version routed-sgbm-pt --param playing_time=playing_time --param league=h2h`.
2. Generate ZAR-reformed and SGP valuations with `use_direct_rates=true` using the PT-adjusted ensemble for both seasons.
3. Run `fbm valuations compare` against three baselines: (a) steamer-only holdout, (b) direct-rate with steamer IP (the current `direct-rate` version), (c) the previous best.
4. Evaluate with `--check` flag; examine pitcher WAR ρ, hit rates, and value MAE.
5. Inspect the top-10 pitcher valuations to verify the distribution looks reasonable (no artificial inflation at the top).
6. Record go/no-go decision with metrics.

### Acceptance criteria

- Side-by-side evaluation results documented for both holdout seasons, for both ZAR-reformed and SGP.
- Pitcher WAR ρ does not regress by more than 0.01 vs. the best available baseline on either season.
- Top-10 pitcher valuations do not show extreme concentration (top pitcher < 2x the 10th pitcher).
- Go/no-go decision recorded with key metrics.
- If go: update production valuation config to use PT-model IP + direct rates with routed ensemble.

## Ordering

Phases are strictly sequential: phase 1 defines targets, phase 2 builds the plumbing, phase 3 does the tuning, phase 4 validates. Phase 1 is pure analysis (no code changes). Phase 2 is a small code change in the ensemble model. Phase 3 is the bulk of the work and may take multiple iterations. Phase 4 is operational (CLI commands + documentation).

This roadmap depends on the direct-rate-stats roadmap (phase 1 done, phase 2 blocked on ensemble fix). The ensemble routing bug must be resolved before phase 2 of this roadmap can produce valid results.
