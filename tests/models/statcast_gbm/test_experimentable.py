"""Tests verifying _StatcastGBMBase satisfies the Experimentable protocol."""

from unittest.mock import MagicMock

from fantasy_baseball_manager.domain.model_protocol import Experimentable, TrainingBackend
from fantasy_baseball_manager.models.statcast_gbm.model import StatcastGBMModel


class TestStatcastGBMExperimentable:
    def test_isinstance_check(self) -> None:
        assembler = MagicMock()
        evaluator = MagicMock()
        model = StatcastGBMModel(assembler, evaluator)
        assert isinstance(model, Experimentable)

    def test_experiment_player_types(self) -> None:
        model = StatcastGBMModel(MagicMock(), MagicMock())
        assert model.experiment_player_types() == ["batter", "pitcher"]

    def test_experiment_feature_columns_returns_lists(self) -> None:
        model = StatcastGBMModel(MagicMock(), MagicMock())
        batter_cols = model.experiment_feature_columns("batter")
        pitcher_cols = model.experiment_feature_columns("pitcher")
        assert isinstance(batter_cols, list)
        assert isinstance(pitcher_cols, list)
        assert len(batter_cols) > 0
        assert len(pitcher_cols) > 0

    def test_experiment_targets_returns_lists(self) -> None:
        model = StatcastGBMModel(MagicMock(), MagicMock())
        batter_targets = model.experiment_targets("batter")
        pitcher_targets = model.experiment_targets("pitcher")
        assert isinstance(batter_targets, list)
        assert isinstance(pitcher_targets, list)
        assert len(batter_targets) > 0
        assert len(pitcher_targets) > 0

    def test_experiment_training_backend_returns_training_backend(self) -> None:
        model = StatcastGBMModel(MagicMock(), MagicMock())
        backend = model.experiment_training_backend()
        assert isinstance(backend, TrainingBackend)

    def test_experiment_training_data_delegates_to_assembler(self) -> None:
        assembler = MagicMock()
        assembler.get_or_materialize.return_value = "handle"
        assembler.read.return_value = [
            {"season": 2023, "col_a": 1.0},
            {"season": 2023, "col_b": 2.0},
            {"season": 2024, "col_c": 3.0},
        ]
        model = StatcastGBMModel(assembler, MagicMock())
        result = model.experiment_training_data("batter", [2023, 2024])
        assert isinstance(result, dict)
        assert 2023 in result
        assert 2024 in result
        assert len(result[2023]) == 2
        assert len(result[2024]) == 1
