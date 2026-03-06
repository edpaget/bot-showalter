from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.state import CompiledStateGraph

from fantasy_baseball_manager.agent.graph import _DEFAULT_MODEL, build_agent
from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.player import Player

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

    from langchain_core.runnables import Runnable


class _FakeChatModel(BaseChatModel):
    """Minimal fake LLM that returns a fixed response and supports bind_tools."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="fake response"))])

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        return self


class _ToolCallingFakeLLM(BaseChatModel):
    """Fake LLM that issues a tool call on first turn, then responds with text."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
        if has_tool_result:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Found the player."))])
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_players",
                    "args": {"name": "Trout"},
                    "id": "call_1",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "tool_calling_fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        return self


class TestBuildAgent:
    def test_returns_compiled_graph(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(SingleConnectionProvider(conn))
        agent = build_agent(container, season=2025, llm=_FakeChatModel())
        assert isinstance(agent, CompiledStateGraph)

    def test_default_model_is_haiku(self) -> None:
        assert _DEFAULT_MODEL == "claude-haiku-4-5-20251001"

    def test_agent_has_tools_bound(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(SingleConnectionProvider(conn))
        agent = build_agent(container, season=2025, llm=_FakeChatModel())
        node_names = set(agent.get_graph().nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names

    def test_agent_processes_simple_message(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(SingleConnectionProvider(conn))
        agent = build_agent(container, season=2025, llm=_FakeChatModel())
        result = agent.invoke({"messages": [("user", "Hello")]})
        messages = result["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1

    def test_agent_executes_tool_and_responds(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(SingleConnectionProvider(conn))
        container.player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
        agent = build_agent(container, season=2025, llm=_ToolCallingFakeLLM())
        result = agent.invoke({"messages": [("user", "Tell me about Trout")]})
        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        assert any("Found the player" in m.content for m in ai_messages)
