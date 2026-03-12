# Flex Position Replacement: Inflated Counts

## Problem

In ZAR valuation, the Hungarian algorithm assigns pitchers to SP, RP, and P (flex) slots. Replacement levels are computed per position as the worst assigned player at that position. Because the best pitchers fill SP/RP slots first, P replacement is much lower, creating a value cliff:

- SP #24 (assigned SP): z=10.77, VAR=0 → **$1**
- SP #25 (assigned P): z=10.59, VAR=4.42 → **$27**

Nearly identical pitchers get 27x different values based solely on which side of the SP/P assignment boundary they land on. This distorts draft strategy — mid-tier SPs are either crushed to $1 (if they make SP) or inflated (if they overflow to P).

## Failed approaches

**Approach 1 — Compare against specific position replacement:** P-assigned SPs use SP replacement for VAR. Mathematically correct but concentrates all surplus in top SPs (Skubal $155 in a $260 league). Equivalent to WAR-style "unified replacement + scarcity premium" — same math, same problem.

**Approach 2 — P replacement = Nth-best in full pool:** Sets P replacement = 48th-best pitcher (for 48 P slots). Too aggressive — marks nearly all P-assigned pitchers as below replacement, losing granularity.

## Solution: Inflated position counts

After the Hungarian assignment, flex-overflow players are counted toward their primary specific position for replacement level computation. The primary position is the eligible specific position with the highest raw replacement (i.e., the scarcest position they could fill).

**Before:** SP replacement = worst of 24 SP-assigned pitchers (high).
**After:** SP replacement = worst of (24 SP-assigned + N P-assigned SPs) (lower).

This eliminates the cliff because SP #24 and SP #25 now compare against the same (inflated) SP replacement level. The total replacement is lower, so more SPs have positive VAR, distributing value smoothly instead of creating a cliff.

### Implementation (assignment.py Step 6)

1. Compute raw specific-position replacements (for primary position tiebreaking).
2. Assign each flex-assigned player a primary specific position (scarcest eligible).
3. Recompute replacement for each specific position including flex-overflow players with that primary.
4. Flex-only players (P-only, UTIL-only) continue using the flex replacement level.
5. VAR for flex-assigned players uses their primary position's inflated replacement.

### Results (2026 keeper league, SP=2 RP=2 P=4, 12 teams)

| Player | Old ($) | New ($) | Notes |
|--------|---------|---------|-------|
| Skubal | 88.0 | 73.3 | Elite SP, slight reduction |
| Crochet | 65.4 | 68.7 | Elite SP, stable |
| Hunter Greene | 8.7 | 44.6 | Was crushed at SP boundary |
| Eovaldi | 5.0 | 35.1 | Was crushed at SP boundary |
| Pivetta | 1.0 | 30.4 | Was $1 replacement level |
| Pablo Lopez (P slot) | 28.2 | 20.4 | P inflation reduced |
| Peralta (P slot) | 27.8 | 19.0 | P inflation reduced |

Top-15 pitcher spread narrowed from $62 ($88→$26) to $46 ($73→$28). Budget totals preserved exactly. Holdout evaluation shows clear improvement on 2024 (WAR ρ 0.20→0.33, hit rates up) and roughly neutral on 2025.

## Inspired by

Smart Fantasy Baseball's "inflated position counts" method (Tanner Bell), which estimates how many players at each primary position fill flex slots and adds those to position totals before computing replacement. Our implementation uses the actual assignment rather than estimates.
