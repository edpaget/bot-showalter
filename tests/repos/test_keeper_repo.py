from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.keeper import KeeperCost
from fantasy_baseball_manager.repos.keeper_repo import SqliteKeeperCostRepo
from tests.helpers import seed_player


class TestSqliteKeeperCostRepo:
    def test_upsert_batch_inserts(self) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteKeeperCostRepo(conn)

        costs = [KeeperCost(player_id=pid, season=2026, league="dynasty", cost=25.0, source="auction")]
        count = repo.upsert_batch(costs)
        conn.commit()

        assert count == 1
        results = repo.find_by_season_league(2026, "dynasty")
        assert len(results) == 1
        assert results[0].player_id == pid
        assert results[0].cost == 25.0
        assert results[0].source == "auction"
        assert results[0].years_remaining == 1
        conn.close()

    def test_upsert_batch_updates_existing(self) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteKeeperCostRepo(conn)

        # Insert initial
        repo.upsert_batch([KeeperCost(player_id=pid, season=2026, league="dynasty", cost=25.0, source="auction")])
        conn.commit()

        # Update with new cost
        repo.upsert_batch(
            [KeeperCost(player_id=pid, season=2026, league="dynasty", cost=30.0, source="contract", years_remaining=2)]
        )
        conn.commit()

        results = repo.find_by_season_league(2026, "dynasty")
        assert len(results) == 1
        assert results[0].cost == 30.0
        assert results[0].source == "contract"
        assert results[0].years_remaining == 2
        conn.close()

    def test_find_by_season_league(self) -> None:
        conn = create_connection(":memory:")
        pid1 = seed_player(conn, name_first="Mike", name_last="Trout")
        pid2 = seed_player(conn, name_first="Shohei", name_last="Ohtani")
        repo = SqliteKeeperCostRepo(conn)

        repo.upsert_batch(
            [
                KeeperCost(player_id=pid1, season=2026, league="dynasty", cost=25.0, source="auction"),
                KeeperCost(player_id=pid2, season=2026, league="dynasty", cost=15.0, source="auction"),
                KeeperCost(player_id=pid1, season=2026, league="other", cost=10.0, source="auction"),
            ]
        )
        conn.commit()

        dynasty = repo.find_by_season_league(2026, "dynasty")
        assert len(dynasty) == 2

        other = repo.find_by_season_league(2026, "other")
        assert len(other) == 1
        assert other[0].player_id == pid1
        conn.close()

    def test_find_by_player(self) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteKeeperCostRepo(conn)

        repo.upsert_batch(
            [
                KeeperCost(player_id=pid, season=2025, league="dynasty", cost=20.0, source="auction"),
                KeeperCost(player_id=pid, season=2026, league="dynasty", cost=25.0, source="auction"),
            ]
        )
        conn.commit()

        results = repo.find_by_player(pid)
        assert len(results) == 2
        seasons = {r.season for r in results}
        assert seasons == {2025, 2026}
        conn.close()

    def test_upsert_with_original_round(self) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteKeeperCostRepo(conn)

        repo.upsert_batch(
            [
                KeeperCost(
                    player_id=pid, season=2026, league="dynasty", cost=18.0, source="draft_round", original_round=3
                )
            ]
        )
        conn.commit()

        results = repo.find_by_season_league(2026, "dynasty")
        assert len(results) == 1
        assert results[0].original_round == 3
        assert results[0].cost == 18.0
        assert results[0].source == "draft_round"
        conn.close()

    def test_upsert_without_original_round(self) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteKeeperCostRepo(conn)

        repo.upsert_batch([KeeperCost(player_id=pid, season=2026, league="dynasty", cost=25.0, source="auction")])
        conn.commit()

        results = repo.find_by_season_league(2026, "dynasty")
        assert len(results) == 1
        assert results[0].original_round is None
        conn.close()

    def test_upsert_updates_original_round(self) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteKeeperCostRepo(conn)

        repo.upsert_batch(
            [
                KeeperCost(
                    player_id=pid, season=2026, league="dynasty", cost=18.0, source="draft_round", original_round=3
                )
            ]
        )
        conn.commit()

        repo.upsert_batch(
            [
                KeeperCost(
                    player_id=pid, season=2026, league="dynasty", cost=22.0, source="draft_round", original_round=2
                )
            ]
        )
        conn.commit()

        results = repo.find_by_season_league(2026, "dynasty")
        assert len(results) == 1
        assert results[0].original_round == 2
        assert results[0].cost == 22.0
        conn.close()

    def test_upsert_batch_idempotent(self) -> None:
        conn = create_connection(":memory:")
        pid = seed_player(conn, name_first="Mike", name_last="Trout")
        repo = SqliteKeeperCostRepo(conn)

        cost = KeeperCost(player_id=pid, season=2026, league="dynasty", cost=25.0, source="auction")

        repo.upsert_batch([cost])
        conn.commit()
        repo.upsert_batch([cost])
        conn.commit()

        results = repo.find_by_season_league(2026, "dynasty")
        assert len(results) == 1
        conn.close()
