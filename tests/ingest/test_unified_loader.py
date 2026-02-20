import sqlite3
from collections.abc import Generator
from typing import Any
from dataclasses import dataclass

import pytest

from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.ingest.loader import Loader
from tests.ingest.conftest import ErrorDataSource, FakeDataSource


@dataclass(frozen=True)
class _FakeRecord:
    name: str


class _FakeRepo:
    def __init__(self) -> None:
        self.upserted: list[_FakeRecord] = []

    def upsert(self, record: _FakeRecord) -> int:
        self.upserted.append(record)
        return len(self.upserted)


class _ErrorRepo:
    def upsert(self, record: Any) -> int:
        raise RuntimeError("upsert exploded")


class _FakeLoadLogRepo:
    def __init__(self) -> None:
        self.logs: list[Any] = []

    def insert(self, log: Any) -> int:
        self.logs.append(log)
        return len(self.logs)

    def get_recent(self, limit: int = 20) -> list[Any]:
        return self.logs[-limit:]

    def get_by_target_table(self, target_table: str) -> list[Any]:
        return [log for log in self.logs if log.target_table == target_table]


def _mapper(row: dict[str, Any]) -> _FakeRecord | None:
    name = row.get("name")
    if name is None:
        return None
    return _FakeRecord(name=name)


@pytest.fixture
def conn() -> Generator[sqlite3.Connection]:
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def log_conn() -> Generator[sqlite3.Connection]:
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


class TestUnifiedLoader:
    def test_success_path_fetch_map_upsert_commit_log(self, conn: sqlite3.Connection) -> None:
        source = FakeDataSource([{"name": "Alice"}, {"name": "Bob"}])
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn)

        result = loader.load()

        assert isinstance(result, Ok)
        log = result.value
        assert log.status == "success"
        assert log.rows_loaded == 2
        assert log.target_table == "widget"
        assert log.source_type == "test"
        assert log.source_detail == "fake"
        assert log.error_message is None
        assert len(repo.upserted) == 2
        assert repo.upserted[0].name == "Alice"
        assert len(log_repo.logs) == 1

    def test_skips_rows_where_mapper_returns_none(self, conn: sqlite3.Connection) -> None:
        source = FakeDataSource([{"name": "Alice"}, {"no_name": True}])
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn)

        result = loader.load()

        assert isinstance(result, Ok)
        assert result.value.rows_loaded == 1
        assert len(repo.upserted) == 1

    def test_fetch_error_writes_error_log_and_returns_err(self, conn: sqlite3.Connection) -> None:
        source = ErrorDataSource()
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn)

        result = loader.load()

        assert isinstance(result, Err)
        assert "fetch failed" in result.error.message
        assert len(log_repo.logs) == 1
        assert log_repo.logs[0].status == "error"
        assert log_repo.logs[0].rows_loaded == 0

    def test_processing_error_rollback_and_error_log(self, conn: sqlite3.Connection) -> None:
        source = FakeDataSource([{"name": "Alice"}])
        repo = _ErrorRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn)

        result = loader.load()

        assert isinstance(result, Err)
        assert "upsert exploded" in result.error.message
        assert len(log_repo.logs) == 1
        assert log_repo.logs[0].status == "error"

    def test_post_upsert_callback_invoked(self, conn: sqlite3.Connection) -> None:
        calls: list[tuple[Any, Any]] = []

        def on_post_upsert(upsert_result: Any, mapped: Any) -> None:
            calls.append((upsert_result, mapped))

        source = FakeDataSource([{"name": "Alice"}, {"name": "Bob"}])
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn, post_upsert=on_post_upsert)

        result = loader.load()

        assert isinstance(result, Ok)
        assert len(calls) == 2
        # upsert_result is the return value of repo.upsert (1-indexed count)
        assert calls[0] == (1, _FakeRecord(name="Alice"))
        assert calls[1] == (2, _FakeRecord(name="Bob"))

    def test_no_callback_by_default(self, conn: sqlite3.Connection) -> None:
        source = FakeDataSource([{"name": "Alice"}])
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn)

        result = loader.load()

        assert isinstance(result, Ok)
        assert result.value.rows_loaded == 1

    def test_log_conn_defaults_to_conn(self, conn: sqlite3.Connection) -> None:
        source = FakeDataSource([{"name": "Alice"}])
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn)

        result = loader.load()

        assert isinstance(result, Ok)
        # If log_conn defaults to conn, both use the same connection.
        # We just verify the load succeeded (no separate conn error).
        assert result.value.status == "success"

    def test_separate_log_conn_receives_log_commits(
        self, conn: sqlite3.Connection, log_conn: sqlite3.Connection
    ) -> None:
        source = FakeDataSource([{"name": "Alice"}])
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn, log_conn=log_conn)

        result = loader.load()

        assert isinstance(result, Ok)
        assert result.value.status == "success"
        assert len(log_repo.logs) == 1

    def test_post_upsert_not_called_for_skipped_rows(self, conn: sqlite3.Connection) -> None:
        calls: list[tuple[Any, Any]] = []

        def on_post_upsert(upsert_result: Any, mapped: Any) -> None:
            calls.append((upsert_result, mapped))

        source = FakeDataSource([{"name": "Alice"}, {"no_name": True}])
        repo = _FakeRepo()
        log_repo = _FakeLoadLogRepo()
        loader = Loader(source, repo, log_repo, _mapper, "widget", conn=conn, post_upsert=on_post_upsert)

        result = loader.load()

        assert isinstance(result, Ok)
        assert len(calls) == 1
        assert calls[0][1].name == "Alice"
