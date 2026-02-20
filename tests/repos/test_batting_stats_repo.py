import sqlite3

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from tests.helpers import seed_player


class TestSqliteBattingStatsRepo:
    def test_upsert_and_get_by_player_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteBattingStatsRepo(conn)
        stats = BattingStats(player_id=player_id, season=2024, source="fangraphs", pa=600, hr=35)
        repo.upsert(stats)
        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].pa == 600
        assert results[0].hr == 35

    def test_get_by_player_season_with_source(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteBattingStatsRepo(conn)
        repo.upsert(BattingStats(player_id=player_id, season=2024, source="fangraphs", pa=600))
        repo.upsert(BattingStats(player_id=player_id, season=2024, source="bbref", pa=598))
        results = repo.get_by_player_season(player_id, 2024, source="fangraphs")
        assert len(results) == 1
        assert results[0].source == "fangraphs"

    def test_get_by_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteBattingStatsRepo(conn)
        repo.upsert(BattingStats(player_id=player_id, season=2024, source="fangraphs"))
        repo.upsert(BattingStats(player_id=player_id, season=2023, source="fangraphs"))
        results = repo.get_by_season(2024)
        assert len(results) == 1

    def test_get_by_season_with_source(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteBattingStatsRepo(conn)
        repo.upsert(BattingStats(player_id=player_id, season=2024, source="fangraphs"))
        repo.upsert(BattingStats(player_id=player_id, season=2024, source="bbref"))
        results = repo.get_by_season(2024, source="bbref")
        assert len(results) == 1
        assert results[0].source == "bbref"

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteBattingStatsRepo(conn)
        repo.upsert(BattingStats(player_id=player_id, season=2024, source="fangraphs", hr=30))
        repo.upsert(BattingStats(player_id=player_id, season=2024, source="fangraphs", hr=35))
        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].hr == 35
