from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from fantasy_baseball_manager.agent.prompt import SYSTEM_PROMPT
from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.tools import create_tools

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def build_agent(
    container: AnalysisContainer,
    *,
    model: str = _DEFAULT_MODEL,
    llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build a ReAct agent wired to the analysis container's tools."""
    if llm is None:
        llm = ChatAnthropic(model=model)  # type: ignore[call-arg] # pragma: no cover
    tools = create_tools(container)
    return create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)
