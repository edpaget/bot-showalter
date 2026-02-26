from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from fantasy_baseball_manager.agent.chat import _print_chunk_content, run_chat

if TYPE_CHECKING:
    import pytest


def _build_fake_agent() -> MagicMock:
    """Build a mock agent that returns a simple response when streamed."""
    agent = MagicMock()
    chunk = MagicMock(spec=AIMessageChunk)
    chunk.content = "Hello back!"
    agent.stream.return_value = [(chunk, {"langgraph_node": "agent"})]
    return agent


class TestRunChat:
    def test_quit_exits_loop(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "quit")
        agent = _build_fake_agent()
        run_chat(agent)
        output = capsys.readouterr().out
        assert "Goodbye!" in output

    def test_exit_exits_loop(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "exit")
        agent = _build_fake_agent()
        run_chat(agent)
        output = capsys.readouterr().out
        assert "Goodbye!" in output

    def test_eof_exits_loop(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        def _raise_eof(prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        agent = _build_fake_agent()
        run_chat(agent)
        output = capsys.readouterr().out
        assert "Goodbye!" in output

    def test_keyboard_interrupt_exits_loop(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def _raise_interrupt(prompt: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_interrupt)
        agent = _build_fake_agent()
        run_chat(agent)
        output = capsys.readouterr().out
        assert "Goodbye!" in output

    def test_empty_input_skipped(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        inputs = iter(["", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent = _build_fake_agent()
        run_chat(agent)
        output = capsys.readouterr().out
        assert "Goodbye!" in output
        agent.stream.assert_not_called()

    def test_streams_ai_message_content(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        inputs = iter(["hello", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent = _build_fake_agent()
        run_chat(agent)
        output = capsys.readouterr().out
        assert "Hello back!" in output

    def test_filters_non_agent_chunks(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        inputs = iter(["hello", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        agent = MagicMock()
        tool_chunk = MagicMock(spec=AIMessageChunk)
        tool_chunk.content = "tool output"
        agent.stream.return_value = [(tool_chunk, {"langgraph_node": "tools"})]
        run_chat(agent)
        output = capsys.readouterr().out
        assert "tool output" not in output


class TestPrintChunkContent:
    def test_prints_string_content(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_chunk_content("hello world")
        output = capsys.readouterr().out
        assert "hello world" in output

    def test_prints_text_block_content(self, capsys: pytest.CaptureFixture[str]) -> None:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": "from block"}]
        _print_chunk_content(blocks)
        output = capsys.readouterr().out
        assert "from block" in output

    def test_skips_non_text_blocks(self, capsys: pytest.CaptureFixture[str]) -> None:
        blocks: list[dict[str, Any]] = [{"type": "tool_use", "name": "search"}]
        _print_chunk_content(blocks)
        output = capsys.readouterr().out
        assert output == ""
