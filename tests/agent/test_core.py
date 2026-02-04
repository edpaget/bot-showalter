"""Integration tests for the agent core."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fantasy_baseball_manager.agent.core import (
    DEFAULT_MODEL,
    Agent,
    _build_system_prompt,
    create_agent,
    run,
)
from fantasy_baseball_manager.valuation.models import (
    LeagueSettings,
    ScoringStyle,
    StatCategory,
)


class TestBuildSystemPrompt:
    def test_includes_league_context(self) -> None:
        mock_settings = LeagueSettings(
            team_count=10,
            batting_categories=(StatCategory.HR, StatCategory.SB),
            pitching_categories=(StatCategory.K, StatCategory.ERA),
            scoring_style=ScoringStyle.H2H_EACH_CATEGORY,
        )

        with patch(
            "fantasy_baseball_manager.agent.core.load_league_settings",
            return_value=mock_settings,
        ):
            prompt = _build_system_prompt()

        assert "10 teams" in prompt
        assert "HR" in prompt
        assert "SB" in prompt
        assert "K" in prompt
        assert "ERA" in prompt

    def test_handles_missing_config(self) -> None:
        with patch(
            "fantasy_baseball_manager.agent.core.load_league_settings",
            side_effect=Exception("Config not found"),
        ):
            prompt = _build_system_prompt()

        assert "not configured" in prompt
        assert "fantasy baseball" in prompt.lower()


class TestCreateAgent:
    def test_creates_agent_with_default_model(self) -> None:
        with patch("fantasy_baseball_manager.agent.core.ChatAnthropic") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            with patch("fantasy_baseball_manager.agent.core.langchain_create_agent") as mock_create:
                mock_graph = MagicMock()
                mock_create.return_value = mock_graph

                agent = create_agent()

        mock_llm_class.assert_called_once_with(model=DEFAULT_MODEL)
        assert isinstance(agent, Agent)
        assert agent.graph == mock_graph
        assert agent.thread_id == "default"

    def test_creates_agent_with_custom_model(self) -> None:
        with patch("fantasy_baseball_manager.agent.core.ChatAnthropic") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            with patch("fantasy_baseball_manager.agent.core.langchain_create_agent") as mock_create:
                mock_graph = MagicMock()
                mock_create.return_value = mock_graph

                _agent = create_agent(model_name="claude-opus-4-20250514")

        mock_llm_class.assert_called_once_with(model="claude-opus-4-20250514")
        assert _agent is not None

    def test_creates_agent_with_custom_llm(self) -> None:
        custom_llm = MagicMock()

        with patch("fantasy_baseball_manager.agent.core.langchain_create_agent") as mock_create:
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            _agent = create_agent(llm=custom_llm)

        # Should use the custom LLM, not create a new one
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == custom_llm
        assert _agent is not None

    def test_creates_agent_with_custom_thread_id(self) -> None:
        with (
            patch("fantasy_baseball_manager.agent.core.ChatAnthropic"),
            patch("fantasy_baseball_manager.agent.core.langchain_create_agent") as mock_create,
        ):
            mock_graph = MagicMock()
            mock_create.return_value = mock_graph

            agent = create_agent(thread_id="custom-thread")

        assert agent.thread_id == "custom-thread"


class TestAgentConfig:
    def test_config_includes_thread_id(self) -> None:
        mock_graph = MagicMock()
        agent = Agent(graph=mock_graph, thread_id="test-thread")

        assert agent.config == {"configurable": {"thread_id": "test-thread"}}


class TestRun:
    def test_invokes_graph_with_message(self) -> None:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [MagicMock(content="Hello there!", __class__=type("AIMessage", (), {}))]
        }

        # Create a proper AIMessage mock
        from langchain_core.messages import AIMessage

        mock_ai_message = AIMessage(content="Test response")
        mock_graph.invoke.return_value = {"messages": [mock_ai_message]}

        agent = Agent(graph=mock_graph, thread_id="test")

        _result = run(agent, "Hello")

        mock_graph.invoke.assert_called_once()
        assert _result is not None
        call_args = mock_graph.invoke.call_args[0][0]
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0].content == "Hello"

    def test_extracts_ai_response(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="What are the top batters?"),
                AIMessage(content="Here are the top batters..."),
            ]
        }

        agent = Agent(graph=mock_graph, thread_id="test")
        result = run(agent, "What are the top batters?")

        assert result == "Here are the top batters..."

    def test_handles_empty_response(self) -> None:
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"messages": []}

        agent = Agent(graph=mock_graph, thread_id="test")
        result = run(agent, "Hello")

        assert "couldn't generate" in result.lower()

    def test_handles_content_blocks(self) -> None:
        from langchain_core.messages import AIMessage

        mock_graph = MagicMock()
        # Some models return content as list of blocks
        mock_graph.invoke.return_value = {"messages": [AIMessage(content=[{"type": "text", "text": "Block response"}])]}

        agent = Agent(graph=mock_graph, thread_id="test")
        result = run(agent, "Hello")

        assert result == "Block response"
