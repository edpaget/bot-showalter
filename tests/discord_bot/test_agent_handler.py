from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from fantasy_baseball_manager.discord_bot.agent_handler import handle_message, split_message


class TestSplitMessage:
    def test_short_text_no_split(self) -> None:
        assert split_message("hello") == ["hello"]

    def test_exact_limit_no_split(self) -> None:
        text = "a" * 2000
        assert split_message(text) == [text]

    def test_paragraph_split(self) -> None:
        first = "a" * 1000
        second = "b" * 1000
        text = first + "\n\n" + second
        result = split_message(text)
        assert result == [first, second]

    def test_newline_fallback(self) -> None:
        first = "a" * 1500
        second = "b" * 1500
        text = first + "\n" + second
        result = split_message(text)
        assert result == [first, second]

    def test_hard_split_at_limit(self) -> None:
        text = "a" * 3000
        result = split_message(text)
        assert result == ["a" * 2000, "a" * 1000]

    def test_empty_string(self) -> None:
        assert split_message("") == []

    def test_custom_limit(self) -> None:
        text = "a" * 10 + "\n\n" + "b" * 10
        result = split_message(text, limit=15)
        assert result == ["a" * 10, "b" * 10]


class TestHandleMessage:
    def test_short_response(self) -> None:
        agent = MagicMock()
        chunk = MagicMock(spec=AIMessageChunk)
        chunk.content = "Short reply."
        agent.stream.return_value = [(chunk, {"langgraph_node": "agent"})]

        result = asyncio.run(handle_message(agent, "hello"))
        assert result == ["Short reply."]

    def test_long_response_splits(self) -> None:
        agent = MagicMock()
        first_para = "a" * 1500
        second_para = "b" * 1500
        chunk = MagicMock(spec=AIMessageChunk)
        chunk.content = first_para + "\n\n" + second_para
        agent.stream.return_value = [(chunk, {"langgraph_node": "agent"})]

        result = asyncio.run(handle_message(agent, "hello"))
        assert result == [first_para, second_para]
