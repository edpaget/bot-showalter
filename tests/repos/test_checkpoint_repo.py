from typing import TYPE_CHECKING

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.checkpoint import FeatureCheckpoint
from fantasy_baseball_manager.domain.experiment import Experiment, TargetResult
from fantasy_baseball_manager.repos.checkpoint_repo import SqliteCheckpointRepo
from fantasy_baseball_manager.repos.errors import DuplicateCheckpointError
from fantasy_baseball_manager.repos.experiment_repo import SqliteExperimentRepo

if TYPE_CHECKING:
    import sqlite3


def _seed_experiment(conn: sqlite3.Connection) -> int:
    """Insert a minimal experiment and return its ID."""
    repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
    return repo.save(
        Experiment(
            timestamp="2026-03-02T12:00:00",
            hypothesis="test",
            model="statcast-gbm-preseason",
            player_type="batter",
            feature_diff={"added": ["barrel_rate"], "removed": []},
            seasons={"train": [2023], "holdout": [2024]},
            params={"n_estimators": 500},
            target_results={
                "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.88),
            },
            conclusion="ok",
        )
    )


def _make_checkpoint(experiment_id: int, **overrides: object) -> FeatureCheckpoint:
    defaults: dict[str, object] = {
        "name": "best_batter_v3",
        "model": "statcast-gbm-preseason",
        "player_type": "batter",
        "feature_columns": ["barrel_rate", "exit_velo"],
        "params": {"n_estimators": 500},
        "target_results": {
            "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.88),
        },
        "experiment_id": experiment_id,
        "created_at": "2026-03-02T12:00:00",
        "notes": "promising set",
    }
    defaults.update(overrides)
    return FeatureCheckpoint(**defaults)  # type: ignore[arg-type]


class TestSqliteCheckpointRepo:
    def test_save_and_get_round_trip(self, conn: sqlite3.Connection) -> None:
        exp_id = _seed_experiment(conn)
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        cp = _make_checkpoint(exp_id)
        repo.save(cp)

        result = repo.get("best_batter_v3", "statcast-gbm-preseason")
        assert result is not None
        assert result.name == "best_batter_v3"
        assert result.model == "statcast-gbm-preseason"
        assert result.player_type == "batter"
        assert result.feature_columns == ["barrel_rate", "exit_velo"]
        assert result.params == {"n_estimators": 500}
        assert result.target_results["slg"].rmse == 0.080
        assert result.target_results["slg"].delta_pct == -5.88
        assert result.experiment_id == exp_id
        assert result.created_at == "2026-03-02T12:00:00"
        assert result.notes == "promising set"

    def test_get_returns_none_when_not_found(self, conn: sqlite3.Connection) -> None:
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        assert repo.get("nonexistent", "m") is None

    def test_duplicate_raises_error(self, conn: sqlite3.Connection) -> None:
        exp_id = _seed_experiment(conn)
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        cp = _make_checkpoint(exp_id)
        repo.save(cp)

        try:
            repo.save(cp)
            msg = "Expected DuplicateCheckpointError"
            raise AssertionError(msg)
        except DuplicateCheckpointError as e:
            assert "best_batter_v3" in str(e)
            assert "statcast-gbm-preseason" in str(e)

    def test_force_overwrite(self, conn: sqlite3.Connection) -> None:
        exp_id = _seed_experiment(conn)
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        cp1 = _make_checkpoint(exp_id, notes="first")
        repo.save(cp1)

        cp2 = _make_checkpoint(exp_id, notes="second")
        repo.save(cp2, force=True)

        result = repo.get("best_batter_v3", "statcast-gbm-preseason")
        assert result is not None
        assert result.notes == "second"

    def test_list_all(self, conn: sqlite3.Connection) -> None:
        exp_id = _seed_experiment(conn)
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        repo.save(_make_checkpoint(exp_id, name="cp1", model="model-a"))
        repo.save(_make_checkpoint(exp_id, name="cp2", model="model-a"))
        repo.save(_make_checkpoint(exp_id, name="cp3", model="model-b"))

        results = repo.list()
        assert len(results) == 3

    def test_list_filter_by_model(self, conn: sqlite3.Connection) -> None:
        exp_id = _seed_experiment(conn)
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        repo.save(_make_checkpoint(exp_id, name="cp1", model="model-a"))
        repo.save(_make_checkpoint(exp_id, name="cp2", model="model-a"))
        repo.save(_make_checkpoint(exp_id, name="cp3", model="model-b"))

        results = repo.list(model="model-a")
        assert len(results) == 2
        assert all(r.model == "model-a" for r in results)

    def test_delete_found(self, conn: sqlite3.Connection) -> None:
        exp_id = _seed_experiment(conn)
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        repo.save(_make_checkpoint(exp_id))

        assert repo.delete("best_batter_v3", "statcast-gbm-preseason") is True
        assert repo.get("best_batter_v3", "statcast-gbm-preseason") is None

    def test_delete_not_found(self, conn: sqlite3.Connection) -> None:
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        assert repo.delete("nonexistent", "m") is False

    def test_same_name_different_models(self, conn: sqlite3.Connection) -> None:
        exp_id = _seed_experiment(conn)
        repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        repo.save(_make_checkpoint(exp_id, name="best", model="model-a"))
        repo.save(_make_checkpoint(exp_id, name="best", model="model-b"))

        assert repo.get("best", "model-a") is not None
        assert repo.get("best", "model-b") is not None
        assert repo.list() == [repo.get("best", "model-a"), repo.get("best", "model-b")]
