import sqlite3

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.tools import create_tools

EXPECTED_TOOL_NAMES = {
    "search_players",
    "get_player_bio",
    "find_players",
    "lookup_projections",
    "lookup_valuations",
    "get_rankings",
    "get_value_over_adp",
    "get_overperformers",
    "get_underperformers",
}


class TestCreateTools:
    def test_returns_nine_tools(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tools = create_tools(container)
        assert len(tools) == 9

    def test_all_tools_have_expected_names(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tools = create_tools(container)
        names = {t.name for t in tools}
        assert names == EXPECTED_TOOL_NAMES

    def test_all_tools_have_description_and_schema(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tools = create_tools(container)
        for t in tools:
            assert t.description, f"Tool {t.name} has no description"
            assert t.args_schema is not None, f"Tool {t.name} has no args_schema"
