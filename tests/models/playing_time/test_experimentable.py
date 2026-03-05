"""Tests verifying PlayingTimeModel satisfies the Experimentable protocol."""

from unittest.mock import MagicMock

from fantasy_baseball_manager.domain.model_protocol import Experimentable, TrainingBackend
from fantasy_baseball_manager.models.playing_time.model import PlayingTimeModel


class TestPlayingTimeExperimentable:
    def test_isinstance_check(self) -> None:
        model = PlayingTimeModel(MagicMock())
        assert isinstance(model, Experimentable)

    def test_experiment_player_types(self) -> None:
        model = PlayingTimeModel(MagicMock())
        assert model.experiment_player_types() == ["batter", "pitcher"]

    def test_experiment_feature_columns_batter(self) -> None:
        model = PlayingTimeModel(MagicMock())
        cols = model.experiment_feature_columns("batter")
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert "age" in cols
        assert "age_pt_factor" in cols

    def test_experiment_feature_columns_pitcher(self) -> None:
        model = PlayingTimeModel(MagicMock())
        cols = model.experiment_feature_columns("pitcher")
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert "age" in cols
        assert "starter_ratio" in cols
        assert "age_pt_factor" in cols

    def test_experiment_targets_batter(self) -> None:
        model = PlayingTimeModel(MagicMock())
        assert model.experiment_targets("batter") == ["pa"]

    def test_experiment_targets_pitcher(self) -> None:
        model = PlayingTimeModel(MagicMock())
        assert model.experiment_targets("pitcher") == ["ip"]

    def test_experiment_training_backend_returns_training_backend(self) -> None:
        model = PlayingTimeModel(MagicMock())
        backend = model.experiment_training_backend()
        assert isinstance(backend, TrainingBackend)

    def test_experiment_training_data_groups_by_season(self) -> None:
        assembler = MagicMock()
        assembler.get_or_materialize.return_value = "handle"
        assembler.read.return_value = [
            {"player_id": 1, "season": 2023, "age": 28, "pa_1": 500, "target_pa": 550},
            {"player_id": 2, "season": 2024, "age": 25, "pa_1": 400, "target_pa": 450},
        ]

        model = PlayingTimeModel(assembler)
        result = model.experiment_training_data("batter", [2023, 2024])

        assert isinstance(result, dict)
        assert 2023 in result
        assert 2024 in result
        assert len(result[2023]) == 1
        assert len(result[2024]) == 1
        # Rows should have age_pt_factor from aging curve enrichment
        assert "age_pt_factor" in result[2023][0]
