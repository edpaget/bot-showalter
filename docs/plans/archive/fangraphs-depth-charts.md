# FanGraphs Depth Charts Import — Roadmap

## Goal

Import FanGraphs Depth Charts projection data from manually downloaded CSV files, storing them as third-party projections alongside existing systems (Marcel, Steamer, ZiPS, etc.). Depth Charts blend Steamer and ZiPS rate projections with human-edited playing time allocations from RosterResource — the industry-standard approach to playing time modeling. This gives us a high-quality playing time baseline without trying to model what is largely a human-judgment problem.

## Background

FanGraphs Depth Charts projections are available via CSV export (requires FanGraphs membership). The export produces separate files for batters and pitchers. The CSV format matches the same FanGraphs projection export format already consumed by our `import` command — the columns use the same names as Steamer/ZiPS exports (`PA`, `HR`, `AVG`, `OBP`, `SLG`, `WAR`, `PlayerId`, `MLBAMID`, etc.).

The existing `import` CLI command already handles FanGraphs projection CSVs using `make_fg_projection_batting_mapper` / `make_fg_projection_pitching_mapper`. However, it treats all imports identically — there's no special handling for Depth Charts' role as a playing time authority.

## Phase 1 — Verify Existing Import Path

**Goal:** Confirm that FanGraphs Depth Charts CSVs import cleanly through the existing `import` command with no code changes.

1. Download Depth Charts CSV exports from FanGraphs (batters and pitchers) for the current season.
2. Import using the existing command:
   ```
   fbm import fangraphs-dc data/fangraphs-dc-batters-2026.csv \
     --version 2026-02-16 --player-type batter --season 2026
   fbm import fangraphs-dc data/fangraphs-dc-pitchers-2026.csv \
     --version 2026-02-16 --player-type pitcher --season 2026
   ```
3. Verify row counts and spot-check a few players via `fbm lookup`.
4. Document any column mismatches or mapping gaps.

**Exit criteria:** Depth Charts data loads into the `projection` table with `system = "fangraphs-dc"` and `source_type = "third_party"`.

## Phase 2 — Playing Time Extraction

**Goal:** Extract PA/IP from Depth Charts projections for use as the playing time baseline in downstream models.

The Depth Charts CSV already includes `PA` (batters) and `IP` (pitchers) as columns, which map to `pa` and `ip` in `stat_json`. No new schema is needed — the data is already stored. This phase adds a service that reads playing time from Depth Charts projections and makes it available to other models.

1. **`DepthChartPlayingTime` protocol** — Define a protocol in `models/playing_time/protocols.py`:
   ```python
   class PlayingTimeSource(Protocol):
       def get_pa(self, player_id: int, season: int) -> float | None: ...
       def get_ip(self, player_id: int, season: int) -> float | None: ...
   ```

2. **`ProjectionPlayingTimeSource`** — Implement against `ProjectionRepo`, filtering by `system = "fangraphs-dc"` and extracting `pa`/`ip` from `stat_json`. This is a thin read-only adapter, not a new model.

3. **Tests** — Verify extraction returns correct PA/IP for known players; returns `None` for players not in Depth Charts.

**Exit criteria:** Other models can inject `PlayingTimeSource` to get Depth Charts PA/IP.

## Phase 3 — Marcel Integration

**Goal:** Let Marcel use Depth Charts playing time instead of (or in addition to) its native formula.

Marcel currently reads playing time projections from the DB when available (`models/marcel/model.py`). This phase wires the Depth Charts playing time into Marcel's pipeline so that Depth Charts PA/IP is used for players it covers, falling back to Marcel's native formula for players without Depth Charts coverage.

1. **Fallback chain** — Create a `FallbackPlayingTimeSource` that tries Depth Charts first, then falls back to the existing playing time model or Marcel's native formula.

2. **Config** — Add a `playing_time_source` option to `fbm.toml` under `[models.marcel.params]`:
   ```toml
   [models.marcel.params]
   playing_time_source = "fangraphs-dc"  # or "native" or "playing-time-model"
   ```

3. **Tests** — Marcel uses Depth Charts PA when available; falls back gracefully; config option works.

**Exit criteria:** `fbm predict marcel --season 2026` uses Depth Charts playing time for covered players.

## Phase 4 — Ensemble Integration

**Goal:** Include Depth Charts as a component in the ensemble model.

The ensemble model (`models/ensemble/`) already blends multiple projection systems via weighted average. Depth Charts can participate as another system.

1. **Config** — Add `fangraphs-dc` to ensemble components in `fbm.toml`:
   ```toml
   [models.ensemble.params.components]
   marcel = 0.5
   "statcast-gbm" = 0.3
   "fangraphs-dc" = 0.2
   ```

2. **Rate vs. PT blending** — The ensemble engine's `blend_rates` function already handles rate/PT separation. Verify it works correctly when one component (Depth Charts) is authoritative on playing time while others are authoritative on rates.

3. **Tests** — Ensemble includes Depth Charts; weights apply correctly; missing players handled.

**Exit criteria:** `fbm predict ensemble --season 2026` incorporates Depth Charts projections.

## Phase 5 — Evaluation

**Goal:** Measure accuracy of Depth Charts playing time against actuals.

1. Import historical Depth Charts CSVs (2024, 2025 preseason snapshots if available).
2. Run `fbm evaluate fangraphs-dc/<version> --season <year>` to get RMSE/MAE for PA and IP against actuals.
3. Compare against Marcel's native playing time and our playing time model.
4. Document results.

**Exit criteria:** Quantitative comparison of Depth Charts PA/IP accuracy vs. alternatives.

## Out of Scope

- **Automated FanGraphs scraping.** CSVs are manually downloaded. Automating this would require dealing with authentication and rate limits.
- **In-season updates.** This is a preseason workflow. In-season re-imports can be done manually with a new `--version` date.
- **Depth chart roster modeling.** We consume the result of RosterResource's depth chart work, not replicate it.
- **Pitcher role classification.** Depth Charts already encodes SP vs. RP in the GS/G ratio; we don't need separate role modeling.
