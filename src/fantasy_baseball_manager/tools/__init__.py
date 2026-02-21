from __future__ import annotations

from langchain_core.tools import BaseTool

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.tools.adp_tools import create_get_value_over_adp_tool
from fantasy_baseball_manager.tools.performance_tools import (
    create_get_overperformers_tool,
    create_get_underperformers_tool,
)
from fantasy_baseball_manager.tools.player_tools import (
    create_find_players_tool,
    create_get_player_bio_tool,
    create_search_players_tool,
)
from fantasy_baseball_manager.tools.projection_tools import create_lookup_projections_tool
from fantasy_baseball_manager.tools.valuation_tools import create_get_rankings_tool, create_lookup_valuations_tool


def create_tools(container: AnalysisContainer) -> list[BaseTool]:
    """Create all LangChain tools wired to the given container."""
    return [
        create_search_players_tool(container),
        create_get_player_bio_tool(container),
        create_find_players_tool(container),
        create_lookup_projections_tool(container),
        create_lookup_valuations_tool(container),
        create_get_rankings_tool(container),
        create_get_value_over_adp_tool(container),
        create_get_overperformers_tool(container),
        create_get_underperformers_tool(container),
    ]
