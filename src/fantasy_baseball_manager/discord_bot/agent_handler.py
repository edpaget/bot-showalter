from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fantasy_baseball_manager.agent.stream import collect_agent_response

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def split_message(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks that fit within Discord's message limit.

    Splits at paragraph boundaries (``\\n\\n``) first, then newlines,
    then hard-splits at the limit.
    """
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    # Try paragraph split
    chunks = _split_on(text, "\n\n", limit)
    if chunks is not None:
        return chunks

    # Fallback to newline split
    chunks = _split_on(text, "\n", limit)
    if chunks is not None:
        return chunks

    # Hard split
    result: list[str] = []
    while text:
        result.append(text[:limit])
        text = text[limit:]
    return result


def _split_on(text: str, separator: str, limit: int) -> list[str] | None:
    """Try splitting text on a separator, returning None if any part exceeds the limit."""
    parts = text.split(separator)
    stripped = [p for p in parts if p]
    if all(len(p) <= limit for p in stripped):
        return stripped
    return None


async def handle_message(agent: CompiledStateGraph, text: str) -> list[str]:
    """Run the agent on *text* and return the response split for Discord."""
    response = await asyncio.to_thread(_run_agent_sync, agent, text)
    return split_message(response)


def _run_agent_sync(agent: CompiledStateGraph, text: str) -> str:
    """Invoke the agent synchronously and collect the full response."""
    stream = agent.stream(
        {"messages": [("user", text)]},
        stream_mode="messages",
    )
    return collect_agent_response(stream)
