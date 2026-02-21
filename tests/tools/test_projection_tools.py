import sqlite3

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.tools.projection_tools import create_lookup_projections_tool
from tests.helpers import seed_player


def _seed_projection(
    conn: sqlite3.Connection,
    player_id: int,
    season: int = 2025,
    system: str = "steamer",
    version: str = "v2025.1",
    player_type: str = "batter",
    stats: dict | None = None,
) -> None:
    repo = SqliteProjectionRepo(conn)
    repo.upsert(
        Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type=player_type,
            stat_json=stats or {"hr": 35, "avg": 0.280, "_mode": "mean"},
        )
    )


class TestLookupProjections:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_lookup_projections_tool(container)
        assert tool.name == "lookup_projections"
        assert tool.description
        assert tool.args_schema is not None

    def test_returns_projection_stats(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_projection(conn, pid, stats={"hr": 35, "avg": 0.280, "_mode": "mean"})
        container = AnalysisContainer(conn)
        tool = create_lookup_projections_tool(container)

        result = tool.run({"player_name": "Soto", "season": 2025})
        assert "Juan Soto" in result
        assert "steamer" in result
        assert "hr: 35" in result
        assert "avg: 0.280" in result
        # Metadata key starting with _ should be filtered
        assert "_mode" not in result

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_lookup_projections_tool(container)

        result = tool.run({"player_name": "Nobody", "season": 2025})
        assert "No projections found" in result
