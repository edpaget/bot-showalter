from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from fantasy_baseball_manager.agent.stream import collect_agent_response, extract_text


class TestExtractText:
    def test_string_content(self) -> None:
        assert extract_text("hello world") == "hello world"

    def test_block_list_with_text(self) -> None:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": "from block"}]
        assert extract_text(blocks) == "from block"

    def test_block_list_skips_non_text(self) -> None:
        blocks: list[dict[str, Any]] = [
            {"type": "tool_use", "name": "search"},
            {"type": "text", "text": "visible"},
        ]
        assert extract_text(blocks) == "visible"

    def test_mixed_blocks(self) -> None:
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "first"},
            {"type": "tool_use", "name": "t"},
            {"type": "text", "text": "second"},
        ]
        assert extract_text(blocks) == "firstsecond"

    def test_empty_string(self) -> None:
        assert extract_text("") == ""

    def test_empty_block_list(self) -> None:
        assert extract_text([]) == ""


class TestCollectAgentResponse:
    def test_collects_agent_chunks(self) -> None:
        chunk1 = MagicMock(spec=AIMessageChunk)
        chunk1.content = "Hello "
        chunk2 = MagicMock(spec=AIMessageChunk)
        chunk2.content = "world!"
        stream = [
            (chunk1, {"langgraph_node": "agent"}),
            (chunk2, {"langgraph_node": "agent"}),
        ]
        assert collect_agent_response(stream) == "Hello world!"

    def test_filters_non_agent_nodes(self) -> None:
        agent_chunk = MagicMock(spec=AIMessageChunk)
        agent_chunk.content = "answer"
        tool_chunk = MagicMock(spec=AIMessageChunk)
        tool_chunk.content = "tool output"
        stream = [
            (tool_chunk, {"langgraph_node": "tools"}),
            (agent_chunk, {"langgraph_node": "agent"}),
        ]
        assert collect_agent_response(stream) == "answer"

    def test_filters_non_ai_message_chunks(self) -> None:
        non_ai = MagicMock()  # no spec=AIMessageChunk
        non_ai.content = "not ai"
        stream = [
            (non_ai, {"langgraph_node": "agent"}),
        ]
        assert collect_agent_response(stream) == ""

    def test_empty_stream(self) -> None:
        assert collect_agent_response([]) == ""

    def test_block_list_content(self) -> None:
        chunk = MagicMock(spec=AIMessageChunk)
        chunk.content = [{"type": "text", "text": "block text"}]
        stream = [(chunk, {"langgraph_node": "agent"})]
        assert collect_agent_response(stream) == "block text"
