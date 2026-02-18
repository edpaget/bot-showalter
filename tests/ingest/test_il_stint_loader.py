from typing import Any

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.result import Ok
from fantasy_baseball_manager.ingest.column_maps import make_il_stint_mapper
from fantasy_baseball_manager.ingest.loader import StatsLoader
from fantasy_baseball_manager.repos.il_stint_repo import SqliteILStintRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from tests.ingest.conftest import FakeDataSource


def _seed_player(conn, *, mlbam_id: int = 545361) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=mlbam_id))


def _il_rows(*overrides: dict[str, Any]) -> list[dict[str, Any]]:
    defaults: dict[str, Any] = {
        "transaction_id": 1,
        "mlbam_id": 545361,
        "date": "2024-05-15T00:00:00",
        "effective_date": "2024-05-15T00:00:00",
        "description": "Los Angeles Angels placed CF Mike Trout on the 15-day injured list. Left knee meniscus tear.",
    }
    return [{**defaults, **o} for o in (overrides or [{}])]


class TestILStintLoader:
    def test_end_to_end_placement_and_activation(self, conn) -> None:
        player_id = _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_il_stint_mapper(players, season=2024)

        rows = [
            {
                "transaction_id": 1,
                "mlbam_id": 545361,
                "date": "2024-05-15T00:00:00",
                "effective_date": "2024-05-15T00:00:00",
                "description": "Los Angeles Angels placed CF Mike Trout on the 15-day injured list."
                " Left knee meniscus tear.",
            },
            {
                "transaction_id": 2,
                "mlbam_id": 545361,
                "date": "2024-06-01T00:00:00",
                "effective_date": "2024-06-01T00:00:00",
                "description": "Los Angeles Angels activated CF Mike Trout from the 15-day injured list.",
            },
        ]
        source = FakeDataSource(rows)
        repo = SqliteILStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "il_stint", conn=conn)

        result = loader.load(season=2024)

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 2
        assert log.target_table == "il_stint"

        stints = repo.get_by_player_season(player_id, 2024)
        assert len(stints) == 2
        types = {s.transaction_type for s in stints}
        assert types == {"placement", "activation"}

    def test_non_il_transactions_skipped(self, conn) -> None:
        _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_il_stint_mapper(players, season=2024)

        rows = [
            {
                "transaction_id": 1,
                "mlbam_id": 545361,
                "date": "2024-05-15T00:00:00",
                "effective_date": "2024-05-15T00:00:00",
                "description": "Los Angeles Angels placed CF Mike Trout on the paternity list.",
            },
            {
                "transaction_id": 2,
                "mlbam_id": 545361,
                "date": "2024-06-01T00:00:00",
                "effective_date": "2024-06-01T00:00:00",
                "description": "Los Angeles Angels placed CF Mike Trout on the 15-day injured list. Left knee surgery.",
            },
        ]
        source = FakeDataSource(rows)
        repo = SqliteILStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "il_stint", conn=conn)

        result = loader.load(season=2024)

        assert isinstance(result, Ok)
        assert result.value.status == "success"
        assert result.value.rows_loaded == 1

    def test_unknown_player_skipped(self, conn) -> None:
        _seed_player(conn, mlbam_id=545361)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_il_stint_mapper(players, season=2024)

        rows = [
            {
                "transaction_id": 1,
                "mlbam_id": 999999,
                "date": "2024-05-15T00:00:00",
                "effective_date": "2024-05-15T00:00:00",
                "description": "Team placed LHP Unknown Player on the 15-day injured list. Shoulder strain.",
            },
        ]
        source = FakeDataSource(rows)
        repo = SqliteILStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)
        loader = StatsLoader(source, repo, log_repo, mapper, "il_stint", conn=conn)

        result = loader.load(season=2024)

        assert isinstance(result, Ok)
        assert result.value.status == "success"
        assert result.value.rows_loaded == 0

    def test_upsert_idempotency_via_loader(self, conn) -> None:
        player_id = _seed_player(conn)
        players = SqlitePlayerRepo(conn).all()
        mapper = make_il_stint_mapper(players, season=2024)

        rows = _il_rows()
        source = FakeDataSource(rows)
        repo = SqliteILStintRepo(conn)
        log_repo = SqliteLoadLogRepo(conn)

        loader = StatsLoader(source, repo, log_repo, mapper, "il_stint", conn=conn)
        loader.load(season=2024)
        loader.load(season=2024)

        stints = repo.get_by_player_season(player_id, 2024)
        assert len(stints) == 1
