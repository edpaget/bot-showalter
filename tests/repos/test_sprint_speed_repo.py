import sqlite3

from fantasy_baseball_manager.domain.sprint_speed import SprintSpeed
from fantasy_baseball_manager.repos.sprint_speed_repo import SqliteSprintSpeedRepo


def _make_sprint_speed(**overrides: object) -> SprintSpeed:
    defaults: dict[str, object] = {
        "mlbam_id": 123456,
        "season": 2024,
        "sprint_speed": 28.5,
        "hp_to_1b": 4.2,
        "bolts": 3,
        "competitive_runs": 50,
    }
    defaults.update(overrides)
    return SprintSpeed(**defaults)  # type: ignore[arg-type]


class TestSqliteSprintSpeedRepo:
    def test_upsert_returns_rowid(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteSprintSpeedRepo(statcast_conn)
        row_id = repo.upsert(_make_sprint_speed())
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_upsert_conflict_updates(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteSprintSpeedRepo(statcast_conn)
        repo.upsert(_make_sprint_speed(sprint_speed=27.0))
        repo.upsert(_make_sprint_speed(sprint_speed=28.5))
        row = statcast_conn.execute(
            "SELECT sprint_speed FROM sprint_speed WHERE mlbam_id = 123456 AND season = 2024"
        ).fetchone()
        assert row["sprint_speed"] == 28.5

    def test_count(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteSprintSpeedRepo(statcast_conn)
        repo.upsert(_make_sprint_speed(mlbam_id=1, season=2023))
        repo.upsert(_make_sprint_speed(mlbam_id=2, season=2023))
        assert repo.count() == 2

    def test_upsert_none_optional_fields(self, statcast_conn: sqlite3.Connection) -> None:
        repo = SqliteSprintSpeedRepo(statcast_conn)
        ss = SprintSpeed(mlbam_id=999, season=2024)
        row_id = repo.upsert(ss)
        assert row_id > 0
        row = statcast_conn.execute(
            "SELECT sprint_speed, hp_to_1b, bolts, competitive_runs FROM sprint_speed WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert row["sprint_speed"] is None
        assert row["hp_to_1b"] is None
        assert row["bolts"] is None
        assert row["competitive_runs"] is None
