# Actual Valuation SP/RP Split

Wire pitcher eligibility (SP/RP/P classification) into `compute_actual_valuations` so that actual valuations use the same positional split as projected valuations.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Add `eligibility_provider` parameter and CLI wiring | done (2026-03-07) |

## Phase 1 — Wire eligibility into actual valuations

1. Add optional `eligibility_provider: EligibilityProvider | None` parameter to `compute_actual_valuations`.
2. When `league.pitcher_positions` is defined AND provider is present, use it for pitcher position classification; otherwise fall back to single P pool.
3. Wire `PlayerEligibilityService` in the CLI command.
4. Tests: SP/RP split test and backward-compat fallback test.

### Acceptance Criteria

- [x] `compute_actual_valuations` accepts optional `eligibility_provider`.
- [x] With provider + `pitcher_positions`, pitchers get SP/RP/P positions.
- [x] Without provider, all pitchers get P (backward compat).
- [x] CLI wires `PlayerEligibilityService` into the call.
- [x] All existing tests pass.
