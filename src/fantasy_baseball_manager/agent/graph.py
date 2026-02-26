from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from fantasy_baseball_manager.agent.prompt import build_system_prompt
from fantasy_baseball_manager.tools import create_tools

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from fantasy_baseball_manager.analysis_container import AnalysisContainer

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def build_agent(
    container: AnalysisContainer,
    *,
    season: int,
    model: str = _DEFAULT_MODEL,
    llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build a ReAct agent wired to the analysis container's tools."""
    if llm is None:
        llm = ChatAnthropic(model=model)  # type: ignore[call-arg] # pragma: no cover
    tools = create_tools(container)
    prompt = build_system_prompt(season)
    return create_react_agent(llm, tools=tools, prompt=prompt)
