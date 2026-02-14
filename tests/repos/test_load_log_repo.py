import sqlite3

from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo


class TestSqliteLoadLogRepo:
    def test_insert_and_get_recent(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLoadLogRepo(conn)
        log = LoadLog(
            source_type="pybaseball",
            source_detail="fg_batting_data",
            target_table="batting_stats",
            rows_loaded=150,
            started_at="2025-01-15T10:00:00",
            finished_at="2025-01-15T10:00:05",
            status="success",
        )
        log_id = repo.insert(log)
        assert log_id > 0
        results = repo.get_recent()
        assert len(results) == 1
        assert results[0].source_type == "pybaseball"
        assert results[0].id == log_id

    def test_get_recent_ordered_by_id_desc(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLoadLogRepo(conn)
        repo.insert(
            LoadLog(
                source_type="pybaseball",
                source_detail="first",
                target_table="player",
                rows_loaded=100,
                started_at="2025-01-15T10:00:00",
                finished_at="2025-01-15T10:00:01",
                status="success",
            )
        )
        repo.insert(
            LoadLog(
                source_type="csv",
                source_detail="second",
                target_table="projection",
                rows_loaded=50,
                started_at="2025-01-15T10:01:00",
                finished_at="2025-01-15T10:01:01",
                status="success",
            )
        )
        results = repo.get_recent()
        assert results[0].source_detail == "second"
        assert results[1].source_detail == "first"

    def test_get_recent_respects_limit(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLoadLogRepo(conn)
        for i in range(5):
            repo.insert(
                LoadLog(
                    source_type="pybaseball",
                    source_detail=f"load_{i}",
                    target_table="batting_stats",
                    rows_loaded=i * 10,
                    started_at="2025-01-15T10:00:00",
                    finished_at="2025-01-15T10:00:01",
                    status="success",
                )
            )
        results = repo.get_recent(limit=3)
        assert len(results) == 3

    def test_get_by_target_table(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLoadLogRepo(conn)
        repo.insert(
            LoadLog(
                source_type="pybaseball",
                source_detail="fg_batting_data",
                target_table="batting_stats",
                rows_loaded=150,
                started_at="2025-01-15T10:00:00",
                finished_at="2025-01-15T10:00:01",
                status="success",
            )
        )
        repo.insert(
            LoadLog(
                source_type="csv",
                source_detail="projections.csv",
                target_table="projection",
                rows_loaded=50,
                started_at="2025-01-15T10:01:00",
                finished_at="2025-01-15T10:01:01",
                status="success",
            )
        )
        results = repo.get_by_target_table("batting_stats")
        assert len(results) == 1
        assert results[0].target_table == "batting_stats"

    def test_insert_with_error_message(self, conn: sqlite3.Connection) -> None:
        repo = SqliteLoadLogRepo(conn)
        repo.insert(
            LoadLog(
                source_type="csv",
                source_detail="bad.csv",
                target_table="batting_stats",
                rows_loaded=0,
                started_at="2025-01-15T10:00:00",
                finished_at="2025-01-15T10:00:01",
                status="error",
                error_message="Parse error on line 42",
            )
        )
        results = repo.get_recent()
        assert results[0].status == "error"
        assert results[0].error_message == "Parse error on line 42"
