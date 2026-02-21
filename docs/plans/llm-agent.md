# LLM Agent for Fantasy Baseball Analysis Roadmap

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Shared analysis container and player biography service | done (2026-02-20) |
| 2 | LangChain tool definitions | pending |
| 3 | Agent loop and CLI entry point | pending |

Expose the project's draft and roster analysis capabilities as tools callable by an LLM agent. The agent will answer natural-language questions about player valuation, projection comparison, draft strategy, and player biography (team, age, experience). The tool implementations share their service wiring with the existing CLI so that business logic is defined once and consumed from both surfaces. The agent loop uses LangChain/LangGraph targeting a small model (Claude Haiku) to keep cost and latency low.

## Phase 1: Shared analysis container and player biography service

Extract a shared DI container that wires all analysis services behind `cached_property` accessors, and add a new `PlayerBiographyService` that supports the biographical queries (players on a team, players under a given age, players with N+ years of experience) that the CLI doesn't currently expose.

### Context

The CLI wires services through ~10 separate context managers in `cli/factory.py`, each opening its own DB connection and constructing a narrow slice of repos and services. This works for single-command invocations but doesn't suit an agent session, which needs many services available simultaneously over a long-lived connection. Additionally, there is no service today that can answer "who are the best players on the Yankees" or "show me players under 25" — the repos have the raw data (roster stints, birth dates) but no service layer ties it together.

### Steps

1. Add a `PlayerBiographyService` in `services/` that accepts `PlayerRepo`, `RosterStintRepo`, `BattingStatsRepo`, `PitchingStatsRepo`, and `PositionAppearanceRepo`. Implement methods:
   - `search(name)` — fuzzy-ish player search returning enriched results (name, team, age, positions, bats/throws).
   - `find(*, team, min_age, max_age, min_experience, max_experience, position, season)` — filter players by biographical criteria. Experience is measured in distinct seasons with MLB batting or pitching stats.
   - `get_bio(player_id)` — full biographical detail for one player (birth date, age, bats/throws, current team, positions played, years of experience, career stat totals).
2. Create `AnalysisContainer` in a new `containers.py` module (sibling to `cli/factory.py`, perhaps at the `services/` or top package level) using the `IngestContainer` pattern — a class that takes a `sqlite3.Connection` and exposes services as `@functools.cached_property`. Include: `player_bio_service`, `projection_lookup_service`, `valuation_lookup_service`, `adp_report_service`, `performance_report_service`, and all repos these services need.
3. Refactor the CLI context managers in `factory.py` to delegate to `AnalysisContainer` where possible, so service construction logic lives in one place. The CLI contexts remain (they manage connection lifecycle), but they instantiate `AnalysisContainer` internally rather than manually wiring each repo and service.
4. Write unit tests for `PlayerBiographyService` following the project's constructor-injection pattern with in-memory SQLite.
5. Write integration tests verifying that `AnalysisContainer` properties resolve correctly and that refactored CLI contexts still produce identical results.

### Acceptance criteria

- `PlayerBiographyService.search("Soto")` returns matching players with current team, age, and primary position.
- `PlayerBiographyService.find(team="NYY", season=2025)` returns Yankees players for 2025.
- `PlayerBiographyService.find(max_age=23, season=2025)` returns players aged 23 or under.
- `PlayerBiographyService.find(min_experience=5, season=2025)` returns players with 5+ MLB seasons.
- `AnalysisContainer` wires all analysis services from a single connection.
- CLI contexts delegate to `AnalysisContainer`; all existing CLI tests pass unchanged.

## Phase 2: LangChain tool definitions

Define LangChain-compatible tool functions that wrap the services from Phase 1. Each tool has a clear name, docstring (used as the LLM's tool description), and typed parameter schema so the agent can decide when and how to call it.

### Context

LangChain tools are Python functions decorated with `@tool` (or classes implementing `BaseTool`) that declare their parameters via type annotations or Pydantic models. The LLM sees the tool name, description, and parameter schema and decides which tool to invoke. Tools should be narrow enough that the LLM can reason about when to use each one, and should return plain text (not complex objects) since the LLM must interpret the output.

### Steps

1. Add `langchain` and `langchain-anthropic` as project dependencies.
2. Create a `tools/` subpackage under the main package. Each module defines one or more tool functions.
3. Implement the following tools, each accepting simple parameters (strings, ints, optional filters) and returning formatted text:
   - `search_players(name: str, season: int)` — Search for players by name. Returns a table of matches with name, team, age, primary position.
   - `get_player_bio(player_name: str, season: int)` — Detailed bio for a single player: age, team, bats/throws, positions, experience, and recent season stats summary.
   - `find_players(season: int, team: str | None, min_age: int | None, max_age: int | None, min_experience: int | None, position: str | None)` — Find players matching biographical filters.
   - `lookup_projections(player_name: str, season: int, system: str | None)` — Projections for a player across systems.
   - `lookup_valuations(player_name: str, season: int, system: str | None)` — Fantasy dollar values and category z-scores for a player.
   - `get_rankings(season: int, system: str, player_type: str | None, position: str | None, top: int)` — Top-N fantasy rankings with values, optionally filtered.
   - `get_value_over_adp(season: int, system: str, version: str, player_type: str | None, top: int)` — Buy targets (underpriced vs ADP), avoid list (overpriced), and unranked sleepers.
   - `get_overperformers(system: str, version: str, season: int, player_type: str, top: int)` — Players who most exceeded their projections.
   - `get_underperformers(system: str, version: str, season: int, player_type: str, top: int)` — Players who most underperformed their projections.
4. Each tool function receives the `AnalysisContainer` via closure (bound at agent startup) — the tool definitions are factory functions that accept the container and return the decorated tool.
5. Add a `create_tools(container: AnalysisContainer) -> list` factory that returns all tools wired to a given container.
6. Write unit tests for each tool using in-memory SQLite and seeded data, verifying the tool returns sensible text output for valid inputs and clear error messages for empty results.

### Acceptance criteria

- Each tool is a valid LangChain tool (has `name`, `description`, `args_schema`).
- Tools return human-readable text, not raw Python objects or JSON.
- `create_tools()` returns the full list of tools, all wired to the same container.
- Tools handle "no results found" gracefully with informative messages rather than empty strings or exceptions.
- All tool functions are covered by unit tests.

## Phase 3: Agent loop and CLI entry point

Wire the tools into a LangGraph ReAct agent, configure it for Claude Haiku, and expose it as an `fbm chat` CLI command for interactive use.

### Context

LangGraph provides a `create_react_agent` helper that pairs an LLM with a set of tools in a reason-then-act loop. The agent receives a user message, decides which tool(s) to call, interprets the results, and either calls more tools or responds. Claude Haiku is cost-effective and fast enough for tool-calling with well-structured tool schemas.

### Steps

1. Add `langgraph` as a project dependency.
2. Create an `agent/` subpackage. Implement `build_agent(container, *, model)` that:
   - Instantiates `ChatAnthropic` with the specified model (default `claude-haiku-4-5-20251001`).
   - Calls `create_tools(container)` to get the tool list.
   - Constructs a `create_react_agent` graph with a system prompt explaining the agent's role as a fantasy baseball analyst and the current season context.
   - Returns the compiled graph.
3. Implement a `run_chat(agent)` function that runs an interactive input loop: reads user messages from stdin, streams agent responses to stdout, and exits on `quit`/`exit`/Ctrl-D.
4. Add `fbm chat` CLI command that:
   - Accepts `--data-dir` (default from config) and `--model` (default Haiku).
   - Builds `AnalysisContainer` from the DB, constructs the agent, and enters the chat loop.
   - Requires an `ANTHROPIC_API_KEY` env var; prints a clear error if missing.
5. Write the system prompt. It should explain: the agent is a fantasy baseball analyst, what season it's analyzing, what tools are available and when to use each one, and that it should cite specific numbers from tool results rather than speculating.
6. Write integration tests that verify the agent graph compiles, binds tools correctly, and can process a simple tool-calling turn (mock the LLM response to avoid live API calls in tests).

### Acceptance criteria

- `fbm chat` starts an interactive session and responds to natural-language questions.
- The agent calls appropriate tools in response to questions like "Who are the most valuable shortstops?" or "Compare Soto's projections across systems."
- Responses cite specific numbers from tool results.
- The agent uses Haiku by default; `--model` flag allows overriding.
- Integration tests pass without requiring an API key (LLM is mocked).

## Ordering

Phase 1 → Phase 2 → Phase 3, strictly sequential. Phase 1 establishes the shared infrastructure that Phase 2's tools consume, and Phase 3 wires the tools into an agent loop. Each phase is independently mergeable — Phase 1 improves the CLI's internal architecture even without the agent, and Phase 2's tools could be consumed by other interfaces (e.g., an MCP server) independently of Phase 3's chat loop.
