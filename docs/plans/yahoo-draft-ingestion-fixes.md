# Yahoo Draft Ingestion Fixes Roadmap

The `fbm yahoo draft-live` command replays Yahoo draft picks into the internal draft engine. After the SP/RP and bench-overflow fix (commit bdaa4ae), 292 of 300 picks ingest successfully. The remaining 8 failures fall into two categories:

1. **Player ID mismatches (7 picks)** — Yahoo maps some players to different internal IDs than our valuation system uses (e.g. José Ramírez: Yahoo uses id 9007, valuations are under id 22036; Michael Busch: Yahoo 18185 vs valuations 19681). These players exist in our pool under different IDs.
2. **Unfillable roster overflow (1 pick)** — When a team leaves a position slot empty (e.g. 0 SS picks) but overflows other positions heavily, all 8 BN slots fill before the last pick. A smarter resolver could place overflow picks into any open slot.

## Status

| Phase | Status |
|-------|--------|
| 1 — Player ID reconciliation | done (2026-03-07) |
| 2 — Greedy slot assignment | done (2026-03-07) |
| 3 — Pitcher position slots | done (2026-03-07) |

## Phase 1: Player ID reconciliation

Fix the 7 "not in available pool" failures by reconciling Yahoo player IDs with valuation player IDs at draft board construction time.

### Context

Yahoo's player mapper resolves Yahoo player keys to internal player IDs, but some players have multiple entries in the `player` table (different eras, name variants, or duplicate imports). The draft board is built from valuations which may use a different player ID than Yahoo mapped. For example:

- José Ramírez: Yahoo -> id 9007 (no 2025 valuations), valuations under id 22036
- Luis Castillo: Yahoo -> id 2187 (no 2025 valuations), valuations under id 8259/14302
- Andrés Giménez: Yahoo -> id 9414 (wrong person: Chris Gimenez), valuations under id 12474
- Michael Busch: Yahoo -> id 18185 (Mike Busch, different person), valuations under id 19681
- Shohei Ohtani (Pitcher): Same id 3771 as Ohtani (Batter) — first pick consumes the pool entry, second is "not in available pool"

### Steps

1. Add an ID alias/redirect table or lookup that maps Yahoo-resolved player IDs to the correct valuation player IDs. This could be a simple `yahoo_player_id_override` table or a name-matching fallback in the draft board builder.
2. In `build_yahoo_draft_setup()`, after building the draft board, create a reverse mapping from Yahoo player IDs to board player IDs using name similarity when IDs don't match directly.
3. Apply the mapping in `ingest_yahoo_pick()` before checking the available pool — if the Yahoo ID isn't in the pool, check for a mapped alias.
4. Handle the Ohtani dual-entry case: when a player appears twice in the draft (batter + pitcher), allow both entries if the valuation system has separate batter/pitcher valuations for the same ID.

### Acceptance criteria

- All 5 name-matchable players (Ramírez, Castillo, García, Giménez, Busch) ingest successfully in the 2025 draft replay.
- Ohtani batter and pitcher picks both ingest (or the pitcher pick logs a clear "dual-entry" message rather than a generic "not in pool" warning).
- No regression: existing passing picks continue to work.
- Unit tests cover the alias lookup and name-matching fallback.

## Phase 2: Greedy slot assignment

Replace the fixed fallback chain in `resolve_draft_position()` with a greedy algorithm that can place a pick into any open roster slot, eliminating the 1 remaining overflow failure.

### Context

The current resolver uses a static fallback chain: primary position -> flex (UTIL/P) -> BN. This fails when a team leaves slots empty (e.g. 0 SS picks) but overflows others heavily. Team 2 in the 2025 draft has 25 picks and 25 roster slots (C=1, 1B=1, 2B=1, 3B=1, SS=1, OF=3, UTIL=1, P=8, BN=8) but the 25th pick (Wilyer Abreu, OF) can't fit because all BN slots are consumed by pitcher and batter overflows, while the SS slot sits empty.

A greedy resolver should try: primary -> flex -> BN -> any open slot. The "any open slot" fallback is acceptable for draft tracking since the position assignment is just for roster bookkeeping, not eligibility.

### Steps

1. Extend `resolve_draft_position()` with a final fallback that iterates all roster slots and returns the first one with remaining capacity.
2. Add tests for the edge case: team with 0 SS picks and 25 total picks should place all 25.
3. Verify with the 2025 draft replay that Wilyer Abreu now ingests.

### Acceptance criteria

- `resolve_draft_position()` never returns an unfillable position when any roster slot has capacity.
- Team 2's 25th pick (Wilyer Abreu) ingests successfully in the 2025 replay.
- Full 2025 replay: 299+ of 300 picks ingest (limited only by pool membership, not slot assignment).
- Unit tests cover the "any open slot" fallback path.

## Phase 3: Pitcher position slots

Use `pitcher_positions` from league config to create SP/RP/P roster slots in the draft engine instead of a single generic P slot, so Yahoo SP/RP positions map directly without needing fallback resolution.

### Context

The `[leagues.h2h.pitcher_positions]` config already specifies `sp = 2, rp = 2, p = 4`, but `build_draft_roster_slots()` ignores this and creates a single `P` slot sized to `roster_pitchers` (8). The current SP/RP -> P fallback works but loses information about which pitcher slots are SP vs RP vs flex P. Using the actual pitcher position breakdown would:

- Allow the draft engine to track SP/RP/P slot usage accurately
- Enable the recommendation engine to factor in SP vs RP needs
- Make the REPL `need` command show SP/RP/P breakdown instead of just "P: 8"

### Steps

1. Modify `build_draft_roster_slots()` to use `league.pitcher_positions` when present, creating separate SP, RP, and P slots instead of a single P slot.
2. Update `resolve_draft_position()` to handle the SP -> RP -> P fallback chain when pitcher sub-positions are roster slots.
3. Ensure backward compatibility: if `pitcher_positions` is empty, fall back to the current `roster_pitchers` -> P behavior.
4. Update mock draft bots and recommendation engine to understand SP/RP/P slot distinctions.

### Acceptance criteria

- When `pitcher_positions` is configured, draft roster slots include SP, RP, and P as separate slots.
- `resolve_draft_position("SP", ...)` assigns to SP slot first, then P, then BN.
- `my_needs()` shows separate SP/RP/P counts.
- Backward compatible: leagues without `pitcher_positions` still work with generic P slot.
- No regression in mock draft or recommendation tests.

## Ordering

**Phase 1** (player ID reconciliation) is the highest priority — it fixes 7 of the 8 remaining failures and addresses a data integrity issue that affects all Yahoo draft features.

**Phase 2** (greedy slot assignment) is a small targeted fix for the 1 remaining edge case. It can be done independently of Phase 1.

**Phase 3** (pitcher position slots) is a quality improvement, not a bug fix. It depends on neither Phase 1 nor 2 but should come after them since it changes the roster slot structure that Phases 1-2 operate on.
