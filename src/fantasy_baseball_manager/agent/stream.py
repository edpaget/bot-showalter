from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.messages import AIMessageChunk


def extract_text(content: str | list[dict[str, Any]]) -> str:
    """Extract text from an AIMessageChunk's content.

    Handles both plain-string and block-list formats.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def collect_agent_response(stream: Iterable[Any]) -> str:
    """Iterate a LangGraph message stream, collecting text from agent AIMessageChunks."""
    parts: list[str] = []
    for chunk, metadata in stream:
        if (
            isinstance(chunk, AIMessageChunk)
            and isinstance(metadata, dict)
            and metadata.get("langgraph_node") == "agent"
        ):
            parts.append(extract_text(chunk.content))
    return "".join(parts)
