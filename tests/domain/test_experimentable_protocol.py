"""Tests for the Experimentable protocol."""

from typing import Any

from fantasy_baseball_manager.domain.model_protocol import (
    Experimentable,
    TargetVector,
    TrainingBackend,
)


class _FakeTrainingBackend:
    def extract_features(self, rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
        return []

    def extract_targets(self, rows: list[dict[str, Any]], targets: list[str]) -> dict[str, TargetVector]:
        return {}

    def fit(self, X: list[list[float]], targets: dict[str, TargetVector], params: dict[str, Any]) -> Any:
        return None  # not called in these tests


class _ValidExperimentable:
    def experiment_player_types(self) -> list[str]:
        return ["batter"]

    def experiment_feature_columns(self, player_type: str) -> list[str]:
        return ["col_a"]

    def experiment_targets(self, player_type: str) -> list[str]:
        return ["slg"]

    def experiment_training_data(self, player_type: str, seasons: list[int]) -> dict[int, list[dict[str, Any]]]:
        return {}

    def experiment_training_backend(self) -> TrainingBackend:
        return _FakeTrainingBackend()  # type: ignore[return-value]


class _MissingMethod:
    def experiment_player_types(self) -> list[str]:
        return ["batter"]

    # Missing the other three methods


class TestExperimentableProtocol:
    def test_satisfying_class_passes_isinstance(self) -> None:
        assert isinstance(_ValidExperimentable(), Experimentable)

    def test_non_satisfying_class_fails_isinstance(self) -> None:
        assert not isinstance(_MissingMethod(), Experimentable)

    def test_plain_object_fails_isinstance(self) -> None:
        assert not isinstance(object(), Experimentable)
