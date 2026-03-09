from typing import TYPE_CHECKING

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3


def _make_valuation(player_id: int, **overrides: object) -> Valuation:
    defaults: dict[str, object] = {
        "player_id": player_id,
        "season": 2025,
        "system": "zar",
        "version": "2025.1",
        "projection_system": "steamer",
        "projection_version": "2025.1",
        "player_type": "batter",
        "position": "OF",
        "value": 32.5,
        "rank": 5,
        "category_scores": {"hr": 1.8, "r": 1.2, "rbi": 0.9, "sb": 0.5, "avg": 0.7},
    }
    defaults.update(overrides)
    return Valuation(**defaults)  # type: ignore[arg-type]


class TestSqliteValuationRepo:
    def test_upsert_and_get_by_player_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        val = _make_valuation(player_id)
        repo.upsert(val)
        results = repo.get_by_player_season(player_id, 2025)
        assert len(results) == 1
        assert results[0].player_id == player_id
        assert results[0].system == "zar"
        assert results[0].value == 32.5
        assert results[0].rank == 5
        assert results[0].position == "OF"
        assert results[0].projection_system == "steamer"

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_valuation(player_id, value=30.0, rank=6))
        repo.upsert(_make_valuation(player_id, value=35.0, rank=3))
        results = repo.get_by_player_season(player_id, 2025)
        assert len(results) == 1
        assert results[0].value == 35.0
        assert results[0].rank == 3

    def test_get_by_player_season_with_system(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_valuation(player_id, system="zar", version="v1"))
        repo.upsert(_make_valuation(player_id, system="other", version="v1"))
        results = repo.get_by_player_season(player_id, 2025, system="zar")
        assert len(results) == 1
        assert results[0].system == "zar"

    def test_get_by_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_valuation(player_id, season=2025))
        repo.upsert(_make_valuation(player_id, season=2024, version="2024.1"))
        results = repo.get_by_season(2025)
        assert len(results) == 1
        assert results[0].season == 2025

    def test_get_by_season_with_system(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_valuation(player_id, system="zar", version="v1"))
        repo.upsert(_make_valuation(player_id, system="other", version="v1"))
        results = repo.get_by_season(2025, system="zar")
        assert len(results) == 1
        assert results[0].system == "zar"

    def test_category_scores_round_trip(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        scores = {"hr": 2.1, "r": 1.5, "rbi": 0.8, "sb": -0.3, "avg": 1.1}
        repo.upsert(_make_valuation(player_id, category_scores=scores))
        results = repo.get_by_player_season(player_id, 2025)
        assert results[0].category_scores == scores

    def test_get_empty_results(self, conn: sqlite3.Connection) -> None:
        seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        assert repo.get_by_player_season(999, 2025) == []
        assert repo.get_by_season(2099) == []

    def test_get_by_season_with_version(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_valuation(player_id, version="production"))
        repo.upsert(_make_valuation(player_id, version="experimental", player_type="pitcher"))
        # Filter by version
        results = repo.get_by_season(2025, version="production")
        assert len(results) == 1
        assert results[0].version == "production"
        # No version filter returns both
        all_results = repo.get_by_season(2025)
        assert len(all_results) == 2

    def test_get_by_season_with_system_and_version(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_valuation(player_id, system="zar", version="v1"))
        repo.upsert(_make_valuation(player_id, system="zar", version="v2", player_type="pitcher"))
        repo.upsert(_make_valuation(player_id, system="other", version="v1", player_type="pitcher"))
        results = repo.get_by_season(2025, system="zar", version="v1")
        assert len(results) == 1
        assert results[0].system == "zar"
        assert results[0].version == "v1"

    def test_multiple_player_types(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteValuationRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_valuation(player_id, player_type="batter", position="OF"))
        repo.upsert(_make_valuation(player_id, player_type="pitcher", position="SP"))
        results = repo.get_by_player_season(player_id, 2025)
        assert len(results) == 2
        types = {r.player_type for r in results}
        assert types == {"batter", "pitcher"}
