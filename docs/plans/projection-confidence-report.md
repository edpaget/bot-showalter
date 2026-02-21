# Projection Confidence Report Roadmap

Surface the degree of agreement (or disagreement) across projection systems for each player. Players where Marcel, Statcast GBM, Steamer, and ZiPS converge are safer bets; players with wide spread are high-variance — either upside plays or landmines. This report helps draft-day decisions by quantifying projection uncertainty without requiring interactive model exploration.

This roadmap depends on: projections from multiple systems (done — Steamer, ZiPS, Marcel, Statcast GBM, composite all produce `Projection` records).

## Status

| Phase | Status |
|-------|--------|
| 1 — Cross-system spread calculation | done (2026-02-20) |
| 2 — High-variance player identification | done (2026-02-20) |
| 3 — CLI commands and report output | not started |

## Phase 1: Cross-System Spread Calculation

Compute per-player, per-stat spread metrics across all available projection systems for a given season.

### Context

The project already stores projections from multiple systems in the `projection` table and has a `compare_cmd()` that evaluates systems against actuals. What's missing is a player-level view that shows how much systems disagree on a specific player. A player where Steamer says 35 HR and ZiPS says 15 HR is fundamentally different from one where all systems say 25 HR, even if the composite is the same.

### Steps

1. Define domain types in `src/fantasy_baseball_manager/domain/projection_confidence.py`:
   - `StatSpread` frozen dataclass: `stat: str`, `min_value: float`, `max_value: float`, `mean: float`, `std: float`, `cv: float` (coefficient of variation = std/mean), `systems: dict[str, float]` (system name to projected value).
   - `PlayerConfidence` frozen dataclass: `player_id: int`, `player_name: str`, `player_type: str`, `position: str`, `spreads: list[StatSpread]`, `overall_cv: float` (average CV across key stats), `agreement_level: str` ("high" / "medium" / "low").
   - `ConfidenceReport` frozen dataclass: `season: int`, `systems: list[str]`, `players: list[PlayerConfidence]`.
2. Build `compute_confidence()` in `src/fantasy_baseball_manager/services/projection_confidence.py`:
   - Accepts `projections: list[Projection]`, `league: LeagueSettings`, `player_names: dict[int, str]`, `min_systems: int = 3`.
   - Groups projections by `player_id` and `player_type`.
   - For each player with >= `min_systems` projections, computes `StatSpread` for each league-relevant stat.
   - Computes `overall_cv` as the mean CV across counting stats (rate stats use a different normalization — CV of the numerator, not the rate itself).
   - Assigns `agreement_level` based on `overall_cv` thresholds (low CV = high agreement).
   - Returns `ConfidenceReport` sorted by `overall_cv` descending (most uncertain first).
3. Write tests with synthetic projections: 3 systems agreeing closely should yield "high" agreement; 3 systems diverging widely should yield "low".

### Acceptance criteria

- `compute_confidence()` produces a `ConfidenceReport` with per-stat spreads for each player.
- Players with only 1-2 system projections are excluded (below `min_systems`).
- Rate stats (AVG, ERA, WHIP) use appropriate spread calculation (not raw CV of the rate).
- `agreement_level` thresholds are configurable and documented.

## Phase 2: High-Variance Player Identification

Classify players into actionable buckets: consensus safe picks, upside gambles, and risky avoids — combining projection spread with ADP positioning.

### Context

Raw spread numbers are useful but not actionable alone. A player with high variance who's going in round 15 is a fine upside dart throw; the same variance at pick 10 is a dangerous risk. This phase combines spread with ADP context to produce draft-actionable classifications.

### Steps

1. Add types to `domain/projection_confidence.py`:
   - `VarianceClassification` enum: `SAFE_CONSENSUS`, `UPSIDE_GAMBLE`, `RISKY_AVOID`, `HIDDEN_UPSIDE`, `KNOWN_QUANTITY`.
   - `ClassifiedPlayer` frozen dataclass: `player: PlayerConfidence`, `classification: VarianceClassification`, `adp_rank: int | None`, `value_rank: int`, `risk_reward_score: float` (upside minus downside, ADP-adjusted).
2. Build `classify_variance()` in `services/projection_confidence.py`:
   - Accepts `report: ConfidenceReport`, `valuations: list[Valuation]`, `adp: list[ADP] | None`.
   - `SAFE_CONSENSUS`: high agreement + ADP within 10 picks of value rank.
   - `UPSIDE_GAMBLE`: low agreement + optimistic system projects top-tier value + ADP is late.
   - `RISKY_AVOID`: low agreement + ADP is early (market overvaluing uncertain player).
   - `HIDDEN_UPSIDE`: medium agreement + at least one system projects significantly above ADP.
   - `KNOWN_QUANTITY`: high agreement regardless of ADP (you know what you're getting).
   - Compute `risk_reward_score` as `(max_system_value - adp_expected_value) - (adp_expected_value - min_system_value)` — positive means more upside than downside.
3. Write tests for each classification bucket with synthetic data.

### Acceptance criteria

- Each player with sufficient data receives exactly one `VarianceClassification`.
- `UPSIDE_GAMBLE` players have at least one system projecting significantly above their ADP value.
- `RISKY_AVOID` players have ADP significantly ahead of their median projected value.
- `risk_reward_score` is positive for upside plays and negative for risky avoids.

## Phase 3: CLI Commands and Report Output

Expose confidence analysis through the CLI with filterable, sortable table output.

### Steps

1. Add `fbm report projection-confidence --season <year> --min-systems <n>`:
   - Prints all players sorted by `overall_cv` descending (most uncertain first).
   - Columns: player name, position, overall CV, agreement level, per-system key stat values.
   - Supports `--player-type batter|pitcher` filter.
   - Supports `--agreement high|medium|low` filter.
2. Add `fbm report variance-targets --season <year> --system <system>`:
   - Prints classified players grouped by `VarianceClassification`.
   - Highlights `UPSIDE_GAMBLE` and `HIDDEN_UPSIDE` as draft targets.
   - Highlights `RISKY_AVOID` as players to fade.
   - Includes ADP rank, value rank, and `risk_reward_score`.
3. Add `fbm report system-disagreements --season <year> --player <name>`:
   - Single-player deep dive: shows all system projections side-by-side for every stat.
   - Highlights stats with the widest disagreement.
4. Register commands in `cli/app.py` under the `report` group.

### Acceptance criteria

- `fbm report projection-confidence` prints a readable table with spread metrics.
- `fbm report variance-targets` groups players into actionable classification buckets.
- `fbm report system-disagreements` shows per-system stat comparison for a single player.
- All commands support `--season` filtering and degrade gracefully when fewer than `min_systems` are available.

## Ordering

Phase 1 is independent and can start immediately — it only requires projection records from multiple systems. Phase 2 depends on phase 1 and benefits from ADP data (but works without it — classifications degrade to agreement-only). Phase 3 depends on phases 1-2. This is one of the lowest-effort roadmaps since it leverages existing multi-system projection data with no new ML models or data sources.
