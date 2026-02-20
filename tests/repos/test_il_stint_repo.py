from fantasy_baseball_manager.domain.il_stint import ILStint
from fantasy_baseball_manager.repos.il_stint_repo import SqliteILStintRepo
from tests.helpers import seed_player


class TestSqliteILStintRepo:
    def test_upsert_and_get_by_player_season(self, conn) -> None:
        player_id = seed_player(conn)
        repo = SqliteILStintRepo(conn)
        stint = ILStint(
            player_id=player_id,
            season=2024,
            start_date="2024-05-15",
            il_type="15",
            injury_location="Right elbow inflammation",
            transaction_type="placement",
        )

        stint_id = repo.upsert(stint)

        assert stint_id is not None
        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].player_id == player_id
        assert results[0].season == 2024
        assert results[0].start_date == "2024-05-15"
        assert results[0].il_type == "15"
        assert results[0].injury_location == "Right elbow inflammation"
        assert results[0].transaction_type == "placement"
        assert results[0].id == stint_id

    def test_upsert_idempotency(self, conn) -> None:
        player_id = seed_player(conn)
        repo = SqliteILStintRepo(conn)
        stint = ILStint(
            player_id=player_id,
            season=2024,
            start_date="2024-05-15",
            il_type="15",
            injury_location="Right elbow",
            transaction_type="placement",
        )
        repo.upsert(stint)

        updated = ILStint(
            player_id=player_id,
            season=2024,
            start_date="2024-05-15",
            il_type="15",
            injury_location="Right elbow inflammation",
            transaction_type="placement",
            end_date="2024-06-01",
            days=17,
        )
        repo.upsert(updated)

        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].injury_location == "Right elbow inflammation"
        assert results[0].end_date == "2024-06-01"
        assert results[0].days == 17

    def test_get_by_season(self, conn) -> None:
        p1 = seed_player(conn, mlbam_id=545361)
        p2 = seed_player(conn, mlbam_id=660271)
        repo = SqliteILStintRepo(conn)
        repo.upsert(ILStint(player_id=p1, season=2024, start_date="2024-05-15", il_type="15"))
        repo.upsert(ILStint(player_id=p2, season=2024, start_date="2024-06-01", il_type="10"))
        repo.upsert(ILStint(player_id=p1, season=2023, start_date="2023-07-01", il_type="60"))

        results = repo.get_by_season(2024)
        assert len(results) == 2

        results_2023 = repo.get_by_season(2023)
        assert len(results_2023) == 1
        assert results_2023[0].il_type == "60"

    def test_get_by_player_returns_all_seasons(self, conn) -> None:
        player_id = seed_player(conn)
        repo = SqliteILStintRepo(conn)
        repo.upsert(ILStint(player_id=player_id, season=2023, start_date="2023-04-10", il_type="15"))
        repo.upsert(ILStint(player_id=player_id, season=2024, start_date="2024-05-15", il_type="10"))

        results = repo.get_by_player(player_id)
        assert len(results) == 2
        seasons = {r.season for r in results}
        assert seasons == {2023, 2024}

    def test_get_by_player_empty(self, conn) -> None:
        player_id = seed_player(conn)
        repo = SqliteILStintRepo(conn)

        results = repo.get_by_player(player_id)
        assert results == []
