import sqlite3

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection, StatDistribution
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


def _seed_projection(conn: sqlite3.Connection) -> tuple[int, int]:
    """Seed a player and projection, returning (player_id, projection_id)."""
    player_id = _seed_player(conn)
    repo = SqliteProjectionRepo(conn)
    proj_id = repo.upsert(
        Projection(
            player_id=player_id,
            season=2025,
            system="steamer",
            version="2025.1",
            player_type="batter",
            stat_json={"hr": 30},
        )
    )
    return player_id, proj_id


class TestProjectionDistributions:
    def test_upsert_and_get_distributions(self, conn: sqlite3.Connection) -> None:
        _player_id, proj_id = _seed_projection(conn)
        repo = SqliteProjectionRepo(conn)
        dist = StatDistribution(
            stat="hr",
            p10=15.0,
            p25=22.0,
            p50=30.0,
            p75=38.0,
            p90=45.0,
            mean=30.5,
            std=9.2,
            family="normal",
        )
        repo.upsert_distributions(proj_id, [dist])
        results = repo.get_distributions(proj_id)
        assert len(results) == 1
        d = results[0]
        assert d.stat == "hr"
        assert d.p10 == 15.0
        assert d.p25 == 22.0
        assert d.p50 == 30.0
        assert d.p75 == 38.0
        assert d.p90 == 45.0
        assert d.mean == 30.5
        assert d.std == 9.2
        assert d.family == "normal"

    def test_upsert_distributions_idempotent(self, conn: sqlite3.Connection) -> None:
        _player_id, proj_id = _seed_projection(conn)
        repo = SqliteProjectionRepo(conn)
        dist_v1 = StatDistribution(stat="hr", p10=15.0, p25=22.0, p50=30.0, p75=38.0, p90=45.0)
        repo.upsert_distributions(proj_id, [dist_v1])
        dist_v2 = StatDistribution(stat="hr", p10=16.0, p25=23.0, p50=31.0, p75=39.0, p90=46.0, mean=31.0)
        repo.upsert_distributions(proj_id, [dist_v2])
        results = repo.get_distributions(proj_id)
        assert len(results) == 1
        assert results[0].p10 == 16.0
        assert results[0].mean == 31.0

    def test_get_distributions_empty(self, conn: sqlite3.Connection) -> None:
        _player_id, proj_id = _seed_projection(conn)
        repo = SqliteProjectionRepo(conn)
        results = repo.get_distributions(proj_id)
        assert results == []

    def test_get_distributions_multiple_stats(self, conn: sqlite3.Connection) -> None:
        _player_id, proj_id = _seed_projection(conn)
        repo = SqliteProjectionRepo(conn)
        dists = [
            StatDistribution(stat="hr", p10=15.0, p25=22.0, p50=30.0, p75=38.0, p90=45.0),
            StatDistribution(stat="avg", p10=0.220, p25=0.250, p50=0.280, p75=0.310, p90=0.340),
            StatDistribution(stat="rbi", p10=60.0, p25=75.0, p50=90.0, p75=105.0, p90=120.0),
        ]
        repo.upsert_distributions(proj_id, dists)
        results = repo.get_distributions(proj_id)
        assert len(results) == 3
        stats = {d.stat for d in results}
        assert stats == {"hr", "avg", "rbi"}

    def test_get_by_player_season_with_distributions(self, conn: sqlite3.Connection) -> None:
        player_id, proj_id = _seed_projection(conn)
        repo = SqliteProjectionRepo(conn)
        dist = StatDistribution(stat="hr", p10=15.0, p25=22.0, p50=30.0, p75=38.0, p90=45.0)
        repo.upsert_distributions(proj_id, [dist])
        results = repo.get_by_player_season(player_id, 2025, include_distributions=True)
        assert len(results) == 1
        assert results[0].distributions is not None
        assert "hr" in results[0].distributions
        assert results[0].distributions["hr"].p50 == 30.0

    def test_get_by_player_season_without_distributions(self, conn: sqlite3.Connection) -> None:
        player_id, proj_id = _seed_projection(conn)
        repo = SqliteProjectionRepo(conn)
        dist = StatDistribution(stat="hr", p10=15.0, p25=22.0, p50=30.0, p75=38.0, p90=45.0)
        repo.upsert_distributions(proj_id, [dist])
        results = repo.get_by_player_season(player_id, 2025)
        assert len(results) == 1
        assert results[0].distributions is None


class TestPctStatsRoundTrip:
    def test_k_pct_bb_pct_round_trip(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn)
        repo = SqliteProjectionRepo(conn)
        stats = {"k_pct": 0.235, "bb_pct": 0.105, "avg": 0.280}
        repo.upsert(
            Projection(
                player_id=player_id,
                season=2025,
                system="mle",
                version="v1",
                player_type="batter",
                stat_json=stats,
            )
        )
        results = repo.get_by_player_season(player_id, 2025, system="mle")
        assert len(results) == 1
        assert results[0].stat_json["k_pct"] == 0.235
        assert results[0].stat_json["bb_pct"] == 0.105
        assert results[0].stat_json["avg"] == 0.280
