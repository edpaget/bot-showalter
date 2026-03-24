"""Tests verifying BreakoutBustModel satisfies the Experimentable protocol."""

from typing import Any
from unittest.mock import MagicMock

from fantasy_baseball_manager.domain import LabeledSeason, OutcomeLabel
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.model_protocol import Experimentable, TrainingBackend
from fantasy_baseball_manager.models.breakout_bust.model import BreakoutBustModel


class _FakeLabelSource:
    def __init__(self, labels: list[Any] | None = None) -> None:
        self._labels = labels or []

    def get_labels(self, season: int) -> list[Any]:
        return [ls for ls in self._labels if ls.season == season]


class TestBreakoutBustExperimentable:
    def test_isinstance_check(self) -> None:
        model = BreakoutBustModel(MagicMock(), _FakeLabelSource())
        assert isinstance(model, Experimentable)

    def test_experiment_player_types(self) -> None:
        model = BreakoutBustModel(MagicMock(), _FakeLabelSource())
        assert model.experiment_player_types() == ["batter", "pitcher"]

    def test_experiment_feature_columns_returns_lists(self) -> None:
        model = BreakoutBustModel(MagicMock(), _FakeLabelSource())
        batter_cols = model.experiment_feature_columns("batter")
        pitcher_cols = model.experiment_feature_columns("pitcher")
        assert isinstance(batter_cols, list)
        assert isinstance(pitcher_cols, list)
        assert len(batter_cols) > 0
        assert len(pitcher_cols) > 0
        # Should include adp columns
        assert "adp_rank" in batter_cols
        assert "adp_pick" in batter_cols

    def test_experiment_targets(self) -> None:
        model = BreakoutBustModel(MagicMock(), _FakeLabelSource())
        batter_targets = model.experiment_targets("batter")
        pitcher_targets = model.experiment_targets("pitcher")
        assert batter_targets == ["p_breakout", "p_bust"]
        assert pitcher_targets == ["p_breakout", "p_bust"]

    def test_experiment_training_backend_returns_training_backend(self) -> None:
        model = BreakoutBustModel(MagicMock(), _FakeLabelSource())
        backend = model.experiment_training_backend()
        assert isinstance(backend, TrainingBackend)

    def test_experiment_training_data_groups_by_season(self) -> None:
        labels = [
            LabeledSeason(
                player_id=1,
                season=2023,
                label=OutcomeLabel.BREAKOUT,
                player_type=PlayerType.BATTER,
                adp_rank=10,
                adp_pick=10,
                actual_value_rank=5,
                rank_delta=5,
            ),
            LabeledSeason(
                player_id=2,
                season=2024,
                label=OutcomeLabel.NEUTRAL,
                player_type=PlayerType.BATTER,
                adp_rank=20,
                adp_pick=20,
                actual_value_rank=20,
                rank_delta=0,
            ),
        ]
        label_source = _FakeLabelSource(labels)

        assembler = MagicMock()
        assembler.get_or_materialize.return_value = "handle"
        assembler.read.return_value = [
            {"player_id": 1, "season": 2023, "col_a": 1.0},
            {"player_id": 2, "season": 2024, "col_b": 2.0},
        ]

        model = BreakoutBustModel(assembler, label_source)
        result = model.experiment_training_data("batter", [2023, 2024])

        assert isinstance(result, dict)
        assert 2023 in result
        assert 2024 in result
        assert len(result[2023]) == 1
        assert len(result[2024]) == 1
        # Rows should have label column from join
        assert result[2023][0]["label"] == OutcomeLabel.BREAKOUT
