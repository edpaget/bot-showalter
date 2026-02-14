import sqlite3

from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo


def _make_record(**overrides: object) -> ModelRunRecord:
    defaults: dict[str, object] = {
        "system": "marcel",
        "version": "2026.1",
        "config_json": {"weights": [5, 4, 3]},
        "artifact_type": "none",
        "created_at": "2026-02-14T12:00:00",
    }
    defaults.update(overrides)
    return ModelRunRecord(**defaults)  # type: ignore[arg-type]


class TestSqliteModelRunRepo:
    def test_upsert_and_get(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        record = _make_record(
            train_dataset_id=None,
            metrics_json={"rmse": 0.15},
            artifact_path="marcel/2026.1",
            git_commit="abc1234",
            tags_json={"experiment": "baseline"},
        )
        row_id = repo.upsert(record)
        assert row_id > 0

        result = repo.get("marcel", "2026.1")
        assert result is not None
        assert result.system == "marcel"
        assert result.version == "2026.1"
        assert result.config_json == {"weights": [5, 4, 3]}
        assert result.artifact_type == "none"
        assert result.created_at == "2026-02-14T12:00:00"
        assert result.metrics_json == {"rmse": 0.15}
        assert result.artifact_path == "marcel/2026.1"
        assert result.git_commit == "abc1234"
        assert result.tags_json == {"experiment": "baseline"}
        assert result.id == row_id

    def test_upsert_updates_on_conflict(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        repo.upsert(_make_record(config_json={"v": 1}))
        repo.upsert(_make_record(config_json={"v": 2}))

        result = repo.get("marcel", "2026.1")
        assert result is not None
        assert result.config_json == {"v": 2}

    def test_get_returns_none_when_missing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        assert repo.get("nonexistent", "v1") is None

    def test_list_all(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        repo.upsert(_make_record(system="marcel", version="2026.1", created_at="2026-02-14T10:00:00"))
        repo.upsert(_make_record(system="xgb", version="v3", created_at="2026-02-14T12:00:00"))
        repo.upsert(_make_record(system="marcel", version="2026.2", created_at="2026-02-14T11:00:00"))

        results = repo.list()
        assert len(results) == 3
        # Ordered by created_at DESC
        assert results[0].system == "xgb"
        assert results[1].version == "2026.2"
        assert results[2].version == "2026.1"

    def test_list_filtered_by_system(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        repo.upsert(_make_record(system="marcel", version="2026.1"))
        repo.upsert(_make_record(system="xgb", version="v3"))
        repo.upsert(_make_record(system="marcel", version="2026.2"))

        results = repo.list(system="marcel")
        assert len(results) == 2
        assert all(r.system == "marcel" for r in results)

    def test_delete(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        repo.upsert(_make_record())
        assert repo.get("marcel", "2026.1") is not None

        repo.delete("marcel", "2026.1")
        assert repo.get("marcel", "2026.1") is None

    def test_nullable_dataset_ids(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        record = _make_record(
            train_dataset_id=None,
            validation_dataset_id=None,
            holdout_dataset_id=None,
        )
        repo.upsert(record)

        result = repo.get("marcel", "2026.1")
        assert result is not None
        assert result.train_dataset_id is None
        assert result.validation_dataset_id is None
        assert result.holdout_dataset_id is None

    def test_json_round_trip(self, conn: sqlite3.Connection) -> None:
        repo = SqliteModelRunRepo(conn)
        config = {"nested": {"key": [1, 2, 3]}, "flag": True}
        metrics = {"rmse": 0.123, "mae": 0.089}
        tags = {"experiment": "ablation", "run": "42"}

        repo.upsert(
            _make_record(
                config_json=config,
                metrics_json=metrics,
                tags_json=tags,
            )
        )

        result = repo.get("marcel", "2026.1")
        assert result is not None
        assert result.config_json == config
        assert result.metrics_json == metrics
        assert result.tags_json == tags
