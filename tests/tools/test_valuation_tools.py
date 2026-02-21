import sqlite3

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.tools.valuation_tools import create_get_rankings_tool, create_lookup_valuations_tool
from tests.helpers import seed_player


def _seed_valuation(
    conn: sqlite3.Connection,
    player_id: int,
    season: int = 2025,
    system: str = "zar",
    version: str = "v1.0",
    player_type: str = "batter",
    position: str = "OF",
    value: float = 42.50,
    rank: int = 1,
    category_scores: dict[str, float] | None = None,
) -> None:
    repo = SqliteValuationRepo(conn)
    repo.upsert(
        Valuation(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            projection_system="steamer",
            projection_version="v2025.1",
            player_type=player_type,
            position=position,
            value=value,
            rank=rank,
            category_scores=category_scores or {"hr": 2.10, "sb": 0.50},
        )
    )


class TestLookupValuations:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_lookup_valuations_tool(container)
        assert tool.name == "lookup_valuations"
        assert tool.description
        assert tool.args_schema is not None

    def test_returns_valuation_data(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid)
        container = AnalysisContainer(conn)
        tool = create_lookup_valuations_tool(container)

        result = tool.run({"player_name": "Soto", "season": 2025})
        assert "Juan Soto" in result
        assert "zar" in result
        assert "$42.50" in result
        assert "Rank: 1" in result
        assert "hr=2.10" in result

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_lookup_valuations_tool(container)

        result = tool.run({"player_name": "Nobody", "season": 2025})
        assert "No valuations found" in result


class TestGetRankings:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_rankings_tool(container)
        assert tool.name == "get_rankings"
        assert tool.description
        assert tool.args_schema is not None

    def test_ranking_order(self, conn: sqlite3.Connection) -> None:
        pid1 = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid2 = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        _seed_valuation(conn, pid1, value=42.50, rank=1)
        _seed_valuation(conn, pid2, value=38.00, rank=2)
        container = AnalysisContainer(conn)
        tool = create_get_rankings_tool(container)

        result = tool.run({"season": 2025, "system": "zar"})
        # Soto should appear before Judge
        soto_pos = result.index("Juan Soto")
        judge_pos = result.index("Aaron Judge")
        assert soto_pos < judge_pos

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_rankings_tool(container)

        result = tool.run({"season": 2025, "system": "zar"})
        assert "No rankings found" in result
