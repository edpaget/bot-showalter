import sqlite3

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo


def _seed_player(conn: sqlite3.Connection) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))


class TestSqliteProjectionRepo:
    def test_upsert_and_get_by_player_season(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        proj = Projection(
            player_id=player_id,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="batter",
            stat_json={"hr": 30, "avg": 0.280},
        )
        repo.upsert(proj)
        results = repo.get_by_player_season(player_id, 2025)
        assert len(results) == 1
        assert results[0].stat_json["hr"] == 30
        assert results[0].stat_json["avg"] == 0.280
        assert results[0].system == "steamer"

    def test_get_by_player_season_with_system(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30},
            )
        )
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="zips",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 25},
            )
        )
        results = repo.get_by_player_season(player_id, 2025, system="zips")
        assert len(results) == 1
        assert results[0].stat_json["hr"] == 25

    def test_get_by_system_version(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30},
            )
        )
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.2",
                player_type="batter",
                stat_json={"hr": 32},
            )
        )
        results = repo.get_by_system_version("steamer", "2025.1")
        assert len(results) == 1
        assert results[0].stat_json["hr"] == 30

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30},
            )
        )
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 35},
            )
        )
        results = repo.get_by_player_season(player_id, 2025, system="steamer")
        assert len(results) == 1
        assert results[0].stat_json["hr"] == 35

    def test_stat_columns_round_trip(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        stats = {"hr": 30, "avg": 0.280, "war": 5.5, "pa": 600, "bb": 70}
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="custom",
                version="v1",
                player_type="batter",
                stat_json=stats,
            )
        )
        results = repo.get_by_player_season(player_id, 2025)
        for key, value in stats.items():
            assert results[0].stat_json[key] == value

    def test_only_known_stat_columns_stored(self, conn: sqlite3.Connection) -> None:
        """Unknown keys in stat_json are silently dropped by the flat schema."""
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="custom",
                version="v1",
                player_type="batter",
                stat_json={"hr": 30, "unknown_stat": 999},
            )
        )
        results = repo.get_by_player_season(player_id, 2025)
        assert results[0].stat_json["hr"] == 30
        assert "unknown_stat" not in results[0].stat_json

    def test_get_by_season(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30},
            )
        )
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2024,
                system="steamer",
                version="2024.1",
                player_type="batter",
                stat_json={"hr": 28},
            )
        )
        results = repo.get_by_season(2025)
        assert len(results) == 1
        assert results[0].season == 2025

    def test_source_type_defaults_to_first_party(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30},
            )
        )
        results = repo.get_by_player_season(player_id, 2025)
        assert len(results) == 1
        assert results[0].source_type == "first_party"

    def test_source_type_third_party_roundtrip(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30},
                source_type="third_party",
            )
        )
        results = repo.get_by_player_season(player_id, 2025)
        assert len(results) == 1
        assert results[0].source_type == "third_party"

    def test_get_by_season_with_system(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="steamer",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 30},
            )
        )
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="zips",
                version="2025.1",
                player_type="batter",
                stat_json={"hr": 25},
            )
        )
        results = repo.get_by_season(2025, system="steamer")
        assert len(results) == 1
        assert results[0].system == "steamer"
