from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.checkpoint import FeatureCheckpoint
from fantasy_baseball_manager.domain.experiment import Experiment, TargetResult
from fantasy_baseball_manager.exceptions import FbmException
from fantasy_baseball_manager.repos.checkpoint_repo import SqliteCheckpointRepo
from fantasy_baseball_manager.repos.experiment_repo import SqliteExperimentRepo
from fantasy_baseball_manager.services.checkpoint_resolver import (
    is_checkpoint_spec,
    resolve_checkpoint,
)

if TYPE_CHECKING:
    import sqlite3


class TestIsCheckpointSpec:
    def test_valid_spec(self) -> None:
        assert is_checkpoint_spec("checkpoint:best_batter_v3") is True

    def test_invalid_no_prefix(self) -> None:
        assert is_checkpoint_spec("best_batter_v3") is False

    def test_invalid_empty_name(self) -> None:
        assert is_checkpoint_spec("checkpoint:") is False

    def test_invalid_different_prefix(self) -> None:
        assert is_checkpoint_spec("model:something") is False


class TestResolveCheckpoint:
    def test_checkpoint_found(self, conn: sqlite3.Connection) -> None:
        exp_repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
        exp_id = exp_repo.save(
            Experiment(
                timestamp="2026-03-02T12:00:00",
                hypothesis="test",
                model="m",
                player_type="batter",
                feature_diff={"added": ["barrel_rate"], "removed": []},
                seasons={"train": [2023], "holdout": [2024]},
                params={"n_estimators": 500},
                target_results={"slg": TargetResult(rmse=0.08, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.88)},
                conclusion="ok",
            )
        )
        cp_repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))
        cp_repo.save(
            FeatureCheckpoint(
                name="best_v3",
                model="m",
                player_type="batter",
                feature_columns=["barrel_rate"],
                params={"n_estimators": 500},
                target_results={"slg": TargetResult(rmse=0.08, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.88)},
                experiment_id=exp_id,
                created_at="2026-03-02T12:00:00",
            )
        )

        result = resolve_checkpoint(cp_repo, "checkpoint:best_v3", "m")
        assert result.name == "best_v3"
        assert result.feature_columns == ["barrel_rate"]

    def test_checkpoint_not_found_raises_config_error(self, conn: sqlite3.Connection) -> None:
        cp_repo = SqliteCheckpointRepo(SingleConnectionProvider(conn))

        with pytest.raises(FbmException):
            resolve_checkpoint(cp_repo, "checkpoint:nonexistent", "m")
