import sqlite3

from fantasy_baseball_manager.domain.statcast_pitch import StatcastPitch
from fantasy_baseball_manager.repos.statcast_pitch_repo import SqliteStatcastPitchRepo


def _make_pitch(
    *,
    game_pk: int = 718001,
    game_date: str = "2024-06-15",
    batter_id: int = 545361,
    pitcher_id: int = 477132,
    at_bat_number: int = 1,
    pitch_number: int = 1,
    pitch_type: str | None = "FF",
    release_speed: float | None = 95.2,
) -> StatcastPitch:
    return StatcastPitch(
        game_pk=game_pk,
        game_date=game_date,
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        at_bat_number=at_bat_number,
        pitch_number=pitch_number,
        pitch_type=pitch_type,
        release_speed=release_speed,
    )


class TestSqliteStatcastPitchRepo:
    def test_upsert_and_get_by_pitcher_date(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        pitch = _make_pitch()
        repo.upsert(pitch)
        results = repo.get_by_pitcher_date(477132, "2024-06-15")
        assert len(results) == 1
        assert results[0].game_pk == 718001
        assert results[0].pitch_type == "FF"
        assert results[0].release_speed == 95.2

    def test_get_by_batter_date(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        repo.upsert(_make_pitch())
        results = repo.get_by_batter_date(545361, "2024-06-15")
        assert len(results) == 1
        assert results[0].batter_id == 545361

    def test_get_by_game(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        repo.upsert(_make_pitch(at_bat_number=1, pitch_number=1))
        repo.upsert(_make_pitch(at_bat_number=1, pitch_number=2))
        results = repo.get_by_game(718001)
        assert len(results) == 2

    def test_upsert_replaces_on_conflict(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        repo.upsert(_make_pitch(pitch_type="FF"))
        repo.upsert(_make_pitch(pitch_type="SL"))
        results = repo.get_by_game(718001)
        assert len(results) == 1
        assert results[0].pitch_type == "SL"

    def test_upsert_returns_id(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        row_id = repo.upsert(_make_pitch())
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_count(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        assert repo.count() == 0
        repo.upsert(_make_pitch(at_bat_number=1, pitch_number=1))
        repo.upsert(_make_pitch(at_bat_number=1, pitch_number=2))
        assert repo.count() == 2

    def test_upsert_preserves_id_on_conflict(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        id1 = repo.upsert(_make_pitch(pitch_type="FF"))
        id2 = repo.upsert(_make_pitch(pitch_type="SL"))
        assert id1 == id2
        results = repo.get_by_game(718001)
        assert results[0].id == id1

    def test_new_columns_round_trip(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteStatcastPitchRepo(statcast_conn)
        pitch = StatcastPitch(
            game_pk=718001,
            game_date="2024-06-15",
            batter_id=545361,
            pitcher_id=477132,
            at_bat_number=1,
            pitch_number=1,
            hc_x=105.3,
            hc_y=160.2,
            stand="R",
            release_extension=6.3,
        )
        repo.upsert(pitch)
        results = repo.get_by_game(718001)
        assert len(results) == 1
        assert results[0].hc_x == 105.3
        assert results[0].hc_y == 160.2
        assert results[0].stand == "R"
        assert results[0].release_extension == 6.3
