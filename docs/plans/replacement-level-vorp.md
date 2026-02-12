# Replacement-Level VORP Roadmap

## Status: Complete

All five phases have been implemented and validated.

## Overview

Added a replacement-level adjustment stage to the valuation pipeline so batter and pitcher values are comparable on a single draft-ranking scale. VORP (Value Over Replacement Player) subtraction establishes a position-aware baseline where 0.0 = freely available on waivers.

Based on the proposal in `docs/proposals/replacement-level-adjustment.md`.

### Design Decisions

- **Util slots**: Treated as a generic batter pool. After position-specific assignment, the next-best unassigned batters fill Util slots with their own replacement threshold.
- **Bench slots**: Ignored. Only starting lineup slots count toward rostered totals.
- **Per-category adjustment**: Proportional scaling — `adjusted_cat = raw_cat × (adjusted_total / raw_total)` — preserving relative category contributions.

## Phases

### Phase 1: Position assignment and replacement-level calculator [Done]

Core module `valuation/replacement.py` with `ReplacementConfig`, position assignment via scarcity priority, smoothing window, and Util pool handling.

### Phase 2: VORP adjustment function [Done]

`apply_replacement_adjustment()` subtracts replacement-level values with per-category proportional scaling for z-score and composite-only path for ml-ridge.

### Phase 3: Draft-rank CLI integration [Done]

Wired into `draft/cli.py` with `--no-replacement` opt-out flag. Adjustment is ON by default.

### Phase 4: Orchestration and draft-simulate integration [Done]

Applied in `shared/orchestration.py` so `draft-simulate` and keeper workflows also use adjusted values.

### Phase 5: Validation and tuning [Done]

Added `--correlation` and `--smoothing-window` flags to `draft-rank` for validation.

#### Results (composite ADP, Yahoo positions)

| Configuration | zscore (top 50) | zscore (top 200) | ml-ridge (top 50) | ml-ridge (top 200) |
|---|---|---|---|---|
| No replacement | 0.696 | 0.662 | 0.659 | 0.610 |
| With replacement (window=5) | 0.743 | 0.703 | 0.686 | 0.627 |
| **Improvement** | **+0.047** | **+0.041** | **+0.027** | **+0.017** |

Key findings:
- Replacement adjustment improves ADP correlation for both valuation methods
- Catchers rank significantly higher (Cal Raleigh: 10 → 5, William Contreras: 38 → 17)
- Abundant positions (OF, 1B) rank lower as expected
- Pitcher Diff values move closer to 0
- Default smoothing window of 5 is optimal (ties window=7 at top 200, wins at top 50)
- zscore outperforms ml-ridge on ADP correlation (0.743 vs 0.686 at top 50)
