# Yahoo Integration Improvements

## Status

| Phase | Description | Status |
|-------|------------|--------|
| 1 | League renewal chain | done (2026-03-04) |
| 2 | Player name normalization | done (2026-03-04) |
| 3 | "Keep best N" keeper model | done (2026-03-04) |

## Context

Three issues block using `yahoo keeper-decisions` for a "keep best 4" league:

1. **League renewal chain** — Yahoo assigns a new league ID each season. The `renew` field in the API links seasons, but we don't parse or store it.
2. **Player name matching** — `YahooPlayerMapper` does no name normalization, so "Bobby Witt Jr.", "Shohei Ohtani (Batter)", "J.D. Martinez", etc. all fail.
3. **"Keep best N" model** — The keeper system is cost-based (surplus = value - cost). This league requires keeping exactly 4 players with no cost.

## Phase 1: League renewal chain

Parse `renew` from the Yahoo API, persist it, and use it to correctly resolve prior-season league keys. Fix the pre-season empty-roster crash.

## Phase 2: Player name normalization

Extract `_normalize_name()` to a shared `name_utils` module and apply it in `YahooPlayerMapper`.

## Phase 3: "Keep best N" keeper model

Support "pick your best N players, no cost" keeper format via config (`keeper_format = "best_n"`, `max_keepers = 4`) and a new derivation flow.
