# Discord Bot Roadmap

Connect the LangGraph agent to Discord so users can @-mention the bot and get fantasy baseball answers. Reuses `build_agent` / `AnalysisContainer` from `fbm chat`. Secrets are supplied via environment variables and documented in `.envrc.sample`.

## Status

| Phase | Status |
|-------|--------|
| 1 — Discord bot with agent integration | done (2026-02-25) |

## Phase 1: Discord bot with agent integration

Stand up a Discord bot that responds to @-mentions by running the user's message through the LangGraph agent and replying with the result.

### Context

The agent is a compiled LangGraph state graph that accepts `{"messages": [("user", text)]}` and streams `AIMessageChunk` responses. `chat.py` already implements the chunk-filtering logic for extracting text from the stream. Discord expects async I/O, so the synchronous `agent.stream()` call needs to run in a thread executor. Discord messages are capped at 2000 characters, so long responses must be split. The bot should be structured cleanly — Discord transport separated from agent invocation — so slash commands, per-channel history, rate limiting, etc. can be layered on later without restructuring.

### Steps

1. Add `discord.py>=2.4` to `pyproject.toml` dependencies.
2. Update `.envrc.sample` to include `ANTHROPIC_API_KEY` and `FBM_DISCORD_TOKEN`.
3. Create a `discord_bot/` subpackage under the main package with two modules:
   - `bot.py` — A thin `discord.Client` subclass that:
     - Builds the `AnalysisContainer` and agent in `setup_hook`.
     - On `on_message`, ignores bot messages and messages that don't @-mention the bot.
     - Strips the mention from the message text and delegates to the agent handler.
     - Shows a typing indicator while the agent works.
   - `agent_handler.py` — A module with a function that:
     - Accepts the agent graph and user text.
     - Runs `agent.stream()` via `asyncio.to_thread`.
     - Collects the full response text using the same chunk-filtering logic as `chat.py` (extract that into a shared helper in `agent/`).
     - Splits responses exceeding 2000 characters at paragraph boundaries and returns a list of message strings.
4. Add an `fbm discord` CLI command that:
   - Requires `FBM_DISCORD_TOKEN` and `ANTHROPIC_API_KEY` env vars; prints clear errors if either is missing.
   - Accepts `--data-dir` and `--model` options matching `fbm chat`.
   - Constructs and runs the bot.
5. Write unit tests: mock the agent to return canned responses, verify @-mention filtering, verify message splitting at the 2000-char boundary, verify bot messages are ignored. Tests must not require a Discord connection or API key.

### Acceptance criteria

- `uv sync` installs `discord.py` without conflicts.
- `.envrc.sample` documents `ANTHROPIC_API_KEY` and `FBM_DISCORD_TOKEN`.
- `fbm discord` prints a clear error when either env var is missing.
- @-mentioning the bot with a question produces an agent-powered reply.
- Messages not mentioning the bot are ignored.
- Responses longer than 2000 characters are split across multiple messages.
- A typing indicator is shown while the agent processes.
- Agent invocation logic is separated from Discord transport (testable independently).
- All tests pass without a real Discord connection or Anthropic API key.

## Ordering

Single phase. Depends on the LLM Agent roadmap (all phases done). Clean separation between transport and agent invocation means future phases (slash commands, per-channel conversation history, rate limiting, allowed-channels config) can be added without restructuring.
