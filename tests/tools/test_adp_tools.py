import sqlite3

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.tools.adp_tools import create_get_value_over_adp_tool
from tests.helpers import seed_player


def _seed_valuation(
    conn: sqlite3.Connection,
    player_id: int,
    rank: int,
    value: float,
    season: int = 2025,
) -> None:
    repo = SqliteValuationRepo(conn)
    repo.upsert(
        Valuation(
            player_id=player_id,
            season=season,
            system="zar",
            version="v1.0",
            projection_system="steamer",
            projection_version="v2025.1",
            player_type="batter",
            position="OF",
            value=value,
            rank=rank,
            category_scores={"hr": 1.0},
        )
    )


def _seed_adp(
    conn: sqlite3.Connection,
    player_id: int,
    rank: int,
    overall_pick: float,
    season: int = 2025,
) -> None:
    repo = SqliteADPRepo(conn)
    repo.upsert(
        ADP(
            player_id=player_id,
            season=season,
            provider="fantasypros",
            overall_pick=overall_pick,
            rank=rank,
            positions="OF",
        )
    )


class TestGetValueOverADP:
    def test_tool_has_valid_attributes(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_value_over_adp_tool(container)
        assert tool.name == "get_value_over_adp"
        assert tool.description
        assert tool.args_schema is not None

    def test_buy_target_appears(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        # ZAR rank 1, ADP rank 10 -> delta = +9 (buy target)
        _seed_valuation(conn, pid, rank=1, value=42.50)
        _seed_adp(conn, pid, rank=10, overall_pick=10.0)
        container = AnalysisContainer(conn)
        tool = create_get_value_over_adp_tool(container)

        result = tool.run({"season": 2025, "system": "zar", "version": "v1.0"})
        assert "Buy Targets" in result
        assert "Juan Soto" in result

    def test_avoid_list_appears(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Overpaid", name_last="Player", mlbam_id=100001)
        # ZAR rank 10, ADP rank 1 -> delta = -9 (overvalued)
        _seed_valuation(conn, pid, rank=10, value=20.00)
        _seed_adp(conn, pid, rank=1, overall_pick=1.0)
        container = AnalysisContainer(conn)
        tool = create_get_value_over_adp_tool(container)

        result = tool.run({"season": 2025, "system": "zar", "version": "v1.0"})
        assert "Avoid List" in result
        assert "Overpaid Player" in result

    def test_unranked_sleeper_appears(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Hidden", name_last="Gem", mlbam_id=100002)
        # ZAR rank 50 with no ADP -> unranked sleeper
        _seed_valuation(conn, pid, rank=50, value=15.00)
        container = AnalysisContainer(conn)
        tool = create_get_value_over_adp_tool(container)

        result = tool.run({"season": 2025, "system": "zar", "version": "v1.0"})
        assert "Unranked Sleepers" in result
        assert "Hidden Gem" in result

    def test_empty_report(self, conn: sqlite3.Connection) -> None:
        container = AnalysisContainer(conn)
        tool = create_get_value_over_adp_tool(container)

        result = tool.run({"season": 2025, "system": "zar", "version": "v1.0"})
        assert "No value-over-ADP data found" in result
