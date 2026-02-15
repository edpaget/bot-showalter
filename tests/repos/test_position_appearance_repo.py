from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.position_appearance import PositionAppearance
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo


def _seed_player(conn, *, mlbam_id: int = 545361) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=mlbam_id))


class TestSqlitePositionAppearanceRepo:
    def test_upsert_and_get_by_player_season(self, conn) -> None:
        player_id = _seed_player(conn)
        repo = SqlitePositionAppearanceRepo(conn)
        pa = PositionAppearance(
            player_id=player_id,
            season=2024,
            position="CF",
            games=120,
        )

        pa_id = repo.upsert(pa)

        assert pa_id is not None
        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].player_id == player_id
        assert results[0].season == 2024
        assert results[0].position == "CF"
        assert results[0].games == 120
        assert results[0].id == pa_id

    def test_upsert_idempotency(self, conn) -> None:
        player_id = _seed_player(conn)
        repo = SqlitePositionAppearanceRepo(conn)
        repo.upsert(
            PositionAppearance(
                player_id=player_id,
                season=2024,
                position="CF",
                games=100,
            )
        )

        repo.upsert(
            PositionAppearance(
                player_id=player_id,
                season=2024,
                position="CF",
                games=120,
            )
        )

        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 1
        assert results[0].games == 120

    def test_multiple_positions_same_season(self, conn) -> None:
        player_id = _seed_player(conn)
        repo = SqlitePositionAppearanceRepo(conn)
        repo.upsert(
            PositionAppearance(
                player_id=player_id,
                season=2024,
                position="CF",
                games=100,
            )
        )
        repo.upsert(
            PositionAppearance(
                player_id=player_id,
                season=2024,
                position="RF",
                games=30,
            )
        )

        results = repo.get_by_player_season(player_id, 2024)
        assert len(results) == 2
        positions = {r.position for r in results}
        assert positions == {"CF", "RF"}

    def test_get_by_player_returns_all_seasons(self, conn) -> None:
        player_id = _seed_player(conn)
        repo = SqlitePositionAppearanceRepo(conn)
        repo.upsert(
            PositionAppearance(
                player_id=player_id,
                season=2023,
                position="CF",
                games=100,
            )
        )
        repo.upsert(
            PositionAppearance(
                player_id=player_id,
                season=2024,
                position="CF",
                games=120,
            )
        )

        results = repo.get_by_player(player_id)
        assert len(results) == 2
        seasons = {r.season for r in results}
        assert seasons == {2023, 2024}

    def test_get_by_season(self, conn) -> None:
        p1 = _seed_player(conn, mlbam_id=545361)
        p2 = _seed_player(conn, mlbam_id=660271)
        repo = SqlitePositionAppearanceRepo(conn)
        repo.upsert(PositionAppearance(player_id=p1, season=2024, position="CF", games=120))
        repo.upsert(PositionAppearance(player_id=p2, season=2024, position="SS", games=140))
        repo.upsert(PositionAppearance(player_id=p1, season=2023, position="CF", games=100))

        results = repo.get_by_season(2024)
        assert len(results) == 2

        results_2023 = repo.get_by_season(2023)
        assert len(results_2023) == 1
        assert results_2023[0].position == "CF"

    def test_get_by_player_empty(self, conn) -> None:
        player_id = _seed_player(conn)
        repo = SqlitePositionAppearanceRepo(conn)

        results = repo.get_by_player(player_id)
        assert results == []
