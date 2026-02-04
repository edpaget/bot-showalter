"""LangGraph ReAct agent core for the fantasy baseball assistant.

This module provides the agent implementation using LangGraph's create_agent function.
The agent can be used via `run()` for synchronous responses or `stream()` for
streaming token output.

Note: Some type: ignore comments are needed because the langchain/langgraph type stubs
are incomplete. The runtime API is correct; the stubs just don't reflect it accurately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain.agents import create_agent as langchain_create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from fantasy_baseball_manager.agent.tools import ALL_TOOLS
from fantasy_baseball_manager.config import load_league_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


DEFAULT_MODEL = "claude-haiku-4-20250514"


def _build_system_prompt() -> str:
    """Build the system prompt with league context."""
    try:
        settings = load_league_settings()
        batting_cats = ", ".join(cat.value for cat in settings.batting_categories)
        pitching_cats = ", ".join(cat.value for cat in settings.pitching_categories)
        league_context = f"""
League Configuration:
- {settings.team_count} teams
- Scoring: {settings.scoring_style.value}
- Batting categories: {batting_cats}
- Pitching categories: {pitching_cats}
"""
    except Exception:
        league_context = "(League settings not configured)"

    return f"""You are a fantasy baseball assistant helping with draft preparation, \
player evaluation, and keeper decisions.

{league_context}

You have access to tools for:
- Projecting batter and pitcher statistics with z-score valuations
- Looking up individual players by name
- Comparing multiple players side-by-side
- Ranking keeper candidates by surplus value over replacement level
- Viewing current league settings

When answering questions:
- Use the available tools to get current projections and data
- Explain your reasoning and highlight key insights
- Consider position scarcity and league settings when making recommendations
- Be concise but thorough in your analysis

If the user asks about players or projections, use the appropriate tool to get the data \
rather than relying on general knowledge."""


@dataclass
class Agent:
    """Fantasy baseball agent wrapping a LangGraph compiled graph."""

    graph: CompiledStateGraph
    thread_id: str = "default"

    @property
    def config(self) -> dict:
        """Return the config dict for graph invocations."""
        return {"configurable": {"thread_id": self.thread_id}}


def create_agent(
    llm: BaseChatModel | None = None,
    model_name: str = DEFAULT_MODEL,
    thread_id: str = "default",
) -> Agent:
    """Create a new fantasy baseball agent.

    Args:
        llm: Optional LLM instance to use. If not provided, creates a ChatAnthropic
            instance with the specified model_name.
        model_name: Model name to use if llm is not provided. Defaults to claude-sonnet.
        thread_id: Thread ID for conversation memory. Defaults to "default".

    Returns:
        An Agent instance ready to handle messages.
    """
    if llm is None:
        # ChatAnthropic accepts 'model' at runtime; type stubs are incomplete
        llm = ChatAnthropic(model=model_name)  # type: ignore[call-arg]

    # Create checkpointer for conversation memory
    checkpointer = MemorySaver()

    # Build the agent
    graph = langchain_create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=_build_system_prompt(),
        checkpointer=checkpointer,
    )

    return Agent(graph=graph, thread_id=thread_id)


def run(agent: Agent, message: str) -> str:
    """Run the agent synchronously and return the response.

    Args:
        agent: The agent instance to use.
        message: The user's message.

    Returns:
        The agent's response as a string.
    """
    result = agent.graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        # Config dict is compatible with RunnableConfig at runtime
        config=agent.config,  # type: ignore[arg-type]
    )

    # Extract the final AI message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # Content may be a string or a list of content blocks
            if isinstance(msg.content, str):
                return msg.content
            # Handle list of content blocks (text, tool_use, etc.)
            text_parts = []
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                return "\n".join(text_parts)

    return "I couldn't generate a response. Please try again."


async def stream(agent: Agent, message: str) -> AsyncIterator[str]:
    """Stream the agent's response token by token.

    Args:
        agent: The agent instance to use.
        message: The user's message.

    Yields:
        String chunks of the response as they are generated.
    """
    async for event in agent.graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        # Config dict is compatible with RunnableConfig at runtime
        config=agent.config,  # type: ignore[arg-type]
        version="v2",
    ):
        kind = event.get("event", "")

        # Stream tokens from the LLM
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content"):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                yield text
