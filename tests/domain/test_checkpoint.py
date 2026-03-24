from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.checkpoint import FeatureCheckpoint
from fantasy_baseball_manager.domain.experiment import TargetResult
from fantasy_baseball_manager.domain.identity import PlayerType


class TestFeatureCheckpoint:
    def test_field_access(self) -> None:
        cp = FeatureCheckpoint(
            name="best_batter_v3",
            model="statcast-gbm-preseason",
            player_type=PlayerType.BATTER,
            feature_columns=["barrel_rate", "exit_velo"],
            params={"n_estimators": 500},
            target_results={
                "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.88),
            },
            experiment_id=42,
            created_at="2026-03-02T12:00:00",
            notes="promising set",
        )
        assert cp.name == "best_batter_v3"
        assert cp.model == "statcast-gbm-preseason"
        assert cp.player_type == "batter"
        assert cp.feature_columns == ["barrel_rate", "exit_velo"]
        assert cp.params == {"n_estimators": 500}
        assert cp.target_results["slg"].rmse == 0.080
        assert cp.experiment_id == 42
        assert cp.created_at == "2026-03-02T12:00:00"
        assert cp.notes == "promising set"

    def test_frozen_immutability(self) -> None:
        cp = FeatureCheckpoint(
            name="test",
            model="m",
            player_type=PlayerType.BATTER,
            feature_columns=["a"],
            params={},
            target_results={},
            experiment_id=1,
            created_at="2026-03-02T12:00:00",
        )
        with pytest.raises(FrozenInstanceError):
            cp.name = "changed"  # type: ignore[misc]

    def test_notes_default_empty(self) -> None:
        cp = FeatureCheckpoint(
            name="test",
            model="m",
            player_type=PlayerType.BATTER,
            feature_columns=[],
            params={},
            target_results={},
            experiment_id=1,
            created_at="2026-03-02T12:00:00",
        )
        assert cp.notes == ""
