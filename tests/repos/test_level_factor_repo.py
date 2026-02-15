import sqlite3

from fantasy_baseball_manager.domain.level_factor import LevelFactor
from fantasy_baseball_manager.repos.level_factor_repo import SqliteLevelFactorRepo


class TestLevelFactorRepo:
    def test_get_by_level_season(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLevelFactorRepo(conn)

        result = repo.get_by_level_season("AAA", 2024)
        assert result is not None
        assert result.level == "AAA"
        assert result.season == 2024
        assert result.factor == 0.80
        assert result.k_factor == 1.15
        assert result.bb_factor == 0.92
        assert result.iso_factor == 0.85
        assert result.babip_factor == 0.95

    def test_get_by_season(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLevelFactorRepo(conn)

        results = repo.get_by_season(2024)
        assert len(results) == 5
        levels = {r.level for r in results}
        assert levels == {"AAA", "AA", "A+", "A", "ROK"}

    def test_level_factors_ordered_correctly(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLevelFactorRepo(conn)

        results = repo.get_by_season(2024)
        factors_by_level = {r.level: r.factor for r in results}
        assert factors_by_level["AAA"] > factors_by_level["AA"]
        assert factors_by_level["AA"] > factors_by_level["A+"]
        assert factors_by_level["A+"] > factors_by_level["A"]
        assert factors_by_level["A"] > factors_by_level["ROK"]

    def test_component_factors_differ(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLevelFactorRepo(conn)

        result = repo.get_by_level_season("AAA", 2024)
        assert result is not None
        components = {result.k_factor, result.bb_factor, result.iso_factor, result.babip_factor}
        assert len(components) == 4

    def test_upsert_and_retrieve(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLevelFactorRepo(conn)

        custom = LevelFactor(
            level="AAA",
            season=2025,
            factor=0.82,
            k_factor=1.12,
            bb_factor=0.93,
            iso_factor=0.87,
            babip_factor=0.96,
        )
        repo.upsert(custom)
        conn.commit()

        result = repo.get_by_level_season("AAA", 2025)
        assert result is not None
        assert result.factor == 0.82
        assert result.k_factor == 1.12

    def test_seed_data_exists_for_all_seasons(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLevelFactorRepo(conn)

        for season in (2022, 2023, 2024):
            results = repo.get_by_season(season)
            assert len(results) == 5, f"Expected 5 level factors for {season}"
