# Yahoo Architecture Fixes Roadmap

The Yahoo Fantasy integration was built across 5 phases and is functionally complete, but an architecture review flagged violations of the project's principles. This roadmap addresses all 10 violations (4 P1, 6 P5) in three phases, progressing from quick fixes through protocol definitions to the largest refactor: extracting business logic from the CLI layer into services.

## Status

| Phase | Status |
|-------|--------|
| 1 — Quick CLI & package fixes | in progress |
| 2 — Yahoo repo protocols & DI | not started |
| 3 — Extract Yahoo services from CLI | not started |

## Phase 1: Quick CLI & package fixes

Address four independent P5 violations that require minimal code changes and no architectural redesign.

### Context

Several small issues were flagged:
- `_resolve_league_context` returns `tuple[str, Any]` instead of `tuple[str, YahooConfig]` (V3) and lets `YahooConfigError` propagate uncaught (V9, a P1).
- `yahoo_sync` and `yahoo_rosters` hardcode season `2026` while other commands accept `--season` (V11).
- `yahoo/__init__.py` is empty — no consolidated public API per principle 6 (V7).

### Steps

1. **Fix `_resolve_league_context` return type** — change `tuple[str, Any]` to `tuple[str, YahooConfig]` and add the import.
2. **Add error handling to `_resolve_league_context`** — wrap `load_yahoo_config()` and `resolve_default_league()` in `try/except YahooConfigError`, calling `print_error` and raising `typer.Exit(code=1)` to match the pattern used by `yahoo_sync` and `yahoo_rosters`.
3. **Add `--season` to `yahoo_sync`** — add `season: Annotated[int, typer.Option(...)] = 2026` parameter, replace the hardcoded `2026` in `get_game_key()`.
4. **Add `--season` to `yahoo_rosters`** — same pattern; replace hardcoded `2026` in `get_game_key()` and `season=2026` in `fetch_team_roster()`.
5. **Populate `yahoo/__init__.py`** — re-export public symbols: `YahooAuth`, `YahooFantasyClient`, `YahooPlayerMapper`, `YahooDraftSource`, `YahooRosterSource`, `YahooLeagueSource`, `YahooTransactionSource`, `YahooLeagueHistorySource`, `YahooDraftPoller`, `extract_player_data`.

### Acceptance criteria

- `_resolve_league_context` return type is `tuple[str, YahooConfig]` with no `Any`.
- Calling any command that uses `_resolve_league_context` with a bad config dir produces a user-friendly error, not a Python traceback.
- `fbm yahoo sync --season 2025` and `fbm yahoo rosters --season 2025` are valid CLI invocations (no hardcoded season in function bodies).
- `from fantasy_baseball_manager.yahoo import YahooFantasyClient` works (re-export).
- All tests pass; no lint errors.

## Phase 2: Yahoo repo protocols & DI

Define protocols for all six Yahoo repos and decouple `YahooPlayerMapper` and `YahooFantasyClient` from concrete implementations. This is the foundation needed before phase 3 can extract services.

### Context

The project's principle 1 requires modules to depend on protocols, not concrete classes. Every non-Yahoo repo already has a protocol in `repos/protocols.py`. The Yahoo repos have none, forcing `YahooPlayerMapper` to import `SqliteYahooPlayerMapRepo` and `SqlitePlayerRepo` directly (V1/V10). `YahooFantasyClient` imports `default_http_retry` from `fantasy_baseball_manager.ingest`, coupling two sibling infrastructure packages (V8). Tests use `# type: ignore[arg-type]` to inject fakes into `YahooContext` because it uses concrete types.

### Steps

1. **Define Yahoo repo protocols** in `repos/protocols.py` — add `YahooLeagueRepo`, `YahooTeamRepo`, `YahooPlayerMapRepo`, `YahooRosterRepo`, `YahooDraftRepo`, `YahooTransactionRepo` protocol classes mirroring the public methods of each `Sqlite*` implementation.
2. **Update `YahooPlayerMapper`** — change constructor to accept `map_repo: YahooPlayerMapRepo` and `player_repo: PlayerRepo` (both protocols). Update the `TYPE_CHECKING` import block.
3. **Update `YahooContext`** in `cli/factory.py` — change all six Yahoo repo fields from concrete `Sqlite*` types to protocol types. Keep `build_yahoo_context` constructing the concrete classes (it's a composition root).
4. **Remove `# type: ignore[arg-type]`** in `tests/yahoo/test_draft_setup.py` and any other test files that needed type-ignore hacks for Yahoo fakes. Verify fakes satisfy the protocols.
5. **Decouple `yahoo/client.py` from `ingest`** — move `default_http_retry` to a shared utility module (e.g., `fantasy_baseball_manager.http_retry`) that both `ingest` and `yahoo` can import, or accept the retry callable purely via constructor injection (the constructor already accepts `retry` — the issue is the module-level `_DEFAULT_RETRY` constant importing from `ingest`).
6. **Re-export new protocols** from `repos/__init__.py`.

### Acceptance criteria

- `grep -rn "SqliteYahoo" src/fantasy_baseball_manager/yahoo/` returns no hits — yahoo package depends only on protocols.
- `YahooContext` fields use protocol types; `# type: ignore[arg-type]` removed from test_draft_setup.py.
- `grep -rn "from fantasy_baseball_manager.ingest" src/fantasy_baseball_manager/yahoo/` returns no hits.
- All existing tests pass unchanged (protocol conformance is structural).
- `uv run ruff check src tests` clean; `uv run ty check src tests` clean.

## Phase 3: Extract Yahoo services from CLI

Move business logic out of CLI commands into service functions, making sync orchestration, keeper cost derivation, and draft setup accessible from any interaction mode (CLI, agent, HTTP).

### Context

Principle 5 requires CLI commands to only parse input, call services, and format output. Currently, `_sync_league_metadata`, `_sync_transactions`, `_build_yahoo_draft_setup`, and `_derive_and_store_keeper_costs` in `cli/commands/yahoo.py` contain orchestration and decision logic (V5). `YahooLeagueSource.fetch()` returns raw dicts, forcing the CLI to do domain mapping (V2). `test_draft_setup.py` tests a private CLI function because the logic isn't in a service (V6).

### Steps

1. **Refactor `YahooLeagueSource.fetch()`** — return `tuple[YahooLeague, list[YahooTeam]]` instead of `dict[str, Any]`. Move the dict-to-domain-object mapping into the source. Update all callers.
2. **Create `src/fantasy_baseball_manager/services/yahoo_sync.py`** — extract `sync_league_metadata(...)` and `sync_transactions(...)` from the CLI helpers. These functions accept repo protocols and sources, not `YahooContext`. The CLI commands become thin wrappers that construct dependencies from `YahooContext` and call the service.
3. **Create `src/fantasy_baseball_manager/services/yahoo_keeper.py`** (or add to existing keeper service) — extract `derive_and_store_keeper_costs(...)` including the "respect manual overrides" filtering logic.
4. **Create `src/fantasy_baseball_manager/services/yahoo_draft_setup.py`** — extract `build_yahoo_draft_setup(...)` from the CLI. Return the same `YahooDraftSetup` dataclass (move it to the service module).
5. **Update CLI commands** — each command becomes: parse args, open context, call service, format output. No domain logic remains.
6. **Move `test_draft_setup.py`** — retarget tests to the service function. Remove dependency on CLI internals. Add tests for the new sync and keeper services.
7. **Re-export new services** from `services/__init__.py`.

### Acceptance criteria

- `grep -rn "def _sync_league_metadata\|def _sync_transactions\|def _derive_and_store_keeper_costs\|def _build_yahoo_draft_setup" src/fantasy_baseball_manager/cli/` returns no hits — all business logic moved to services.
- `YahooLeagueSource.fetch()` returns `tuple[YahooLeague, list[YahooTeam]]`, not `dict`.
- `test_draft_setup.py` imports from `services`, not `cli.commands.yahoo`.
- New service functions are independently testable with protocol-typed fakes (no `YahooContext` needed in service tests).
- All tests pass; coverage threshold met.

## Ordering

**Phase 1** has no dependencies and can be done immediately. All changes are independent quick fixes.

**Phase 2** has no hard dependency on phase 1 but is logically ordered after it — the protocols it introduces are the foundation for phase 3's service extraction. Phase 2 should be completed before starting phase 3.

**Phase 3** depends on phase 2 — the extracted services need to accept protocol-typed repos, not concrete classes. Phase 3 is the largest phase and could optionally be split into sub-phases (sync services first, then keeper/draft services) if desired.
