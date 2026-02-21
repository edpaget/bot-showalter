import sqlite3

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.tools.performance_tools import (
    create_get_overperformers_tool,
    create_get_underperformers_tool,
)
from tests.helpers import seed_player


def _seed_projection(conn: sqlite3.Connection, player_id: int, stats: dict) -> None:
    repo = SqliteProjectionRepo(conn)
    repo.upsert(
        Projection(
            player_id=player_id,
            season=2025,
            system="steamer",
            version="v2025.1",
            player_type="batter",
            stat_json=stats,
        )
    )


def _seed_batting(conn: sqlite3.Connection, player_id: int, **kwargs: object) -> None:
    repo = SqliteBattingStatsRepo(conn)
    repo.upsert(BattingStats(player_id=player_id, season=2025, source="fangraphs", pa=500, **kwargs))  # type: ignore[arg-type]


class TestGetOverperformers:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_overperformers_tool(container)
        assert tool.name == "get_overperformers"
        assert tool.description
        assert tool.args_schema is not None

    def test_overperformer_sorted_correctly(self, conn: sqlite3.Connection) -> None:
        # Player A: avg projected .250, actual .300 -> delta +.050 (overperformer)
        # Player B: avg projected .270, actual .280 -> delta +.010 (slight overperformer)
        pid_a = seed_player(conn, name_first="Big", name_last="Overperformer", mlbam_id=100001)
        pid_b = seed_player(conn, name_first="Small", name_last="Overperformer", mlbam_id=100002)
        _seed_projection(conn, pid_a, {"avg": 0.250})
        _seed_projection(conn, pid_b, {"avg": 0.270})
        _seed_batting(conn, pid_a, avg=0.300)
        _seed_batting(conn, pid_b, avg=0.280)
        container = AnalysisContainer(conn)
        tool = create_get_overperformers_tool(container)

        result = tool.run(
            {
                "system": "steamer",
                "version": "v2025.1",
                "season": 2025,
                "player_type": "batter",
            }
        )
        assert "Overperformers" in result
        # Big should be listed before Small since higher delta
        big_pos = result.index("Big Overperformer")
        small_pos = result.index("Small Overperformer")
        assert big_pos < small_pos

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_overperformers_tool(container)

        result = tool.run(
            {
                "system": "steamer",
                "version": "v2025.1",
                "season": 2025,
                "player_type": "batter",
            }
        )
        assert "No performance data found" in result


class TestGetUnderperformers:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_underperformers_tool(container)
        assert tool.name == "get_underperformers"
        assert tool.description
        assert tool.args_schema is not None

    def test_underperformer_sorted_correctly(self, conn: sqlite3.Connection) -> None:
        # Player A: avg projected .300, actual .250 -> delta -.050 (underperformer)
        # Player B: avg projected .280, actual .270 -> delta -.010 (slight underperformer)
        pid_a = seed_player(conn, name_first="Big", name_last="Underperformer", mlbam_id=100001)
        pid_b = seed_player(conn, name_first="Small", name_last="Underperformer", mlbam_id=100002)
        _seed_projection(conn, pid_a, {"avg": 0.300})
        _seed_projection(conn, pid_b, {"avg": 0.280})
        _seed_batting(conn, pid_a, avg=0.250)
        _seed_batting(conn, pid_b, avg=0.270)
        container = AnalysisContainer(conn)
        tool = create_get_underperformers_tool(container)

        result = tool.run(
            {
                "system": "steamer",
                "version": "v2025.1",
                "season": 2025,
                "player_type": "batter",
            }
        )
        assert "Underperformers" in result
        # Big underperformer (more negative delta) should be listed first
        big_pos = result.index("Big Underperformer")
        small_pos = result.index("Small Underperformer")
        assert big_pos < small_pos

    def test_no_results(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_underperformers_tool(container)

        result = tool.run(
            {
                "system": "steamer",
                "version": "v2025.1",
                "season": 2025,
                "player_type": "batter",
            }
        )
        assert "No performance data found" in result
