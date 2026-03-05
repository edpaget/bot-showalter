"""Tests for the Experimentable protocol."""

from typing import Any

from fantasy_baseball_manager.domain.model_protocol import Experimentable


class _ValidExperimentable:
    def experiment_player_types(self) -> list[str]:
        return ["batter"]

    def experiment_feature_columns(self, player_type: str) -> list[str]:
        return ["col_a"]

    def experiment_targets(self, player_type: str) -> list[str]:
        return ["slg"]

    def experiment_training_data(self, player_type: str, seasons: list[int]) -> dict[int, list[dict[str, Any]]]:
        return {}


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
