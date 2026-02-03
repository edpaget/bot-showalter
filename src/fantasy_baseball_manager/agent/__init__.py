"""Fantasy baseball LLM agent module.

This module provides an interactive agent that can answer natural-language
fantasy baseball questions using the existing projection, valuation, and
keeper analysis functionality.

Public API:
    create_agent(llm, model_name, thread_id) -> Agent
    run(agent, message) -> str
    stream(agent, message) -> AsyncIterator[str]

Example usage:
    from fantasy_baseball_manager.agent import create_agent, run

    agent = create_agent()
    response = run(agent, "Who are the top 10 projected batters?")
    print(response)
"""

from fantasy_baseball_manager.agent.core import (
    DEFAULT_MODEL,
    Agent,
    create_agent,
    run,
    stream,
)

__all__ = [
    "DEFAULT_MODEL",
    "Agent",
    "create_agent",
    "run",
    "stream",
]
