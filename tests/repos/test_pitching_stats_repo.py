import sqlite3

from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from tests.helpers import seed_player


class TestSqlitePitchingStatsRepo:
    def test_upsert_and_get_by_player_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        repo = SqlitePitchingStatsRepo(conn)
        stats = PitchingStats(player_id=player_id, season=2024, source="fangraphs", w=15, era=2.89)
        repo.upsert(stats)
        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].w == 15
        assert results[0].era == 2.89

    def test_get_by_player_season_with_source(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        repo = SqlitePitchingStatsRepo(conn)
        repo.upsert(PitchingStats(player_id=player_id, season=2024, source="fangraphs", so=250))
        repo.upsert(PitchingStats(player_id=player_id, season=2024, source="bbref", so=248))
        results = repo.get_by_player_season(player_id, 2024, source="bbref")
        assert len(results) == 1
        assert results[0].so == 248

    def test_get_by_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        repo = SqlitePitchingStatsRepo(conn)
        repo.upsert(PitchingStats(player_id=player_id, season=2024, source="fangraphs"))
        repo.upsert(PitchingStats(player_id=player_id, season=2023, source="fangraphs"))
        results = repo.get_by_season(2024)
        assert len(results) == 1

    def test_get_by_season_with_source(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        repo = SqlitePitchingStatsRepo(conn)
        repo.upsert(PitchingStats(player_id=player_id, season=2024, source="fangraphs"))
        repo.upsert(PitchingStats(player_id=player_id, season=2024, source="bbref"))
        results = repo.get_by_season(2024, source="fangraphs")
        assert len(results) == 1

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        repo = SqlitePitchingStatsRepo(conn)
        repo.upsert(PitchingStats(player_id=player_id, season=2024, source="fangraphs", era=3.50))
        repo.upsert(PitchingStats(player_id=player_id, season=2024, source="fangraphs", era=2.89))
        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].era == 2.89
