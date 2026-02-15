from typing import Any

from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.playing_time.model import PlayingTimeModel
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Preparable,
    Trainable,
)


class TestPlayingTimeModelProtocol:
    def test_is_model(self) -> None:
        assert isinstance(PlayingTimeModel(), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(PlayingTimeModel(), Preparable)

    def test_is_predictable(self) -> None:
        assert isinstance(PlayingTimeModel(), Predictable)

    def test_is_not_trainable(self) -> None:
        assert not isinstance(PlayingTimeModel(), Trainable)

    def test_is_not_evaluable(self) -> None:
        assert not isinstance(PlayingTimeModel(), Evaluable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(PlayingTimeModel(), FineTunable)

    def test_name(self) -> None:
        assert PlayingTimeModel().name == "playing_time"

    def test_supported_operations(self) -> None:
        assert PlayingTimeModel().supported_operations == frozenset({"prepare", "predict"})

    def test_artifact_type(self) -> None:
        assert PlayingTimeModel().artifact_type == "none"


class FakeAssembler:
    """In-memory assembler for testing predict()."""

    def __init__(self, batting_rows: list[dict[str, Any]], pitching_rows: list[dict[str, Any]] | None = None) -> None:
        self._batting_rows = batting_rows
        self._pitching_rows = pitching_rows or []
        self._next_id = 1

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        if "pitching" in feature_set.name:
            rows = self._pitching_rows
        else:
            rows = self._batting_rows
        handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=self._next_id,
            table_name=f"ds_{feature_set.name}",
            row_count=len(rows),
            seasons=feature_set.seasons,
        )
        self._next_id += 1
        return handle

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        return DatasetSplits(train=handle, validation=None, holdout=None)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        if "pitching" in handle.table_name:
            return self._pitching_rows
        return self._batting_rows


class TestPlayingTimePredict:
    def test_predict_batter_returns_pa(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        assert result.model_name == "playing_time"
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        assert pred["player_id"] == 1
        assert "pa" in pred
        assert "ip" not in pred or pred.get("ip") is None

    def test_predict_batter_pa_value(self) -> None:
        """PA = baseline(200) + 0.5*600 + 0.1*550 = 200 + 300 + 55 = 555."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        pred = result.predictions[0]
        assert pred["pa"] == 555

    def test_predict_pitcher_returns_ip(self) -> None:
        pitching_rows = [
            {
                "player_id": 10,
                "season": 2023,
                "age": 28,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "g_1": 30,
                "g_2": 28,
                "gs_1": 30,
                "gs_2": 28,
            },
        ]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        pitcher_preds = [p for p in result.predictions if p.get("player_type") == "pitcher"]
        assert len(pitcher_preds) == 1
        assert "ip" in pitcher_preds[0]
        assert "pa" not in pitcher_preds[0] or pitcher_preds[0].get("pa") is None

    def test_predict_pitcher_ip_value_starter(self) -> None:
        """Starter: IP = baseline(60) + 0.5*180 + 0.1*170 = 60 + 90 + 17 = 167."""
        pitching_rows = [
            {
                "player_id": 10,
                "season": 2023,
                "age": 28,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "g_1": 30,
                "g_2": 28,
                "gs_1": 30,
                "gs_2": 28,
            },
        ]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        pitcher_pred = [p for p in result.predictions if p.get("player_type") == "pitcher"][0]
        assert pitcher_pred["ip"] == 167.0

    def test_predict_pitcher_ip_value_reliever(self) -> None:
        """Reliever (gs/g < 0.5): IP = baseline(25) + 0.5*70 + 0.1*65 = 25 + 35 + 6.5 = 66.5."""
        pitching_rows = [
            {
                "player_id": 11,
                "season": 2023,
                "age": 30,
                "ip_1": 70.0,
                "ip_2": 65.0,
                "g_1": 60,
                "g_2": 55,
                "gs_1": 0,
                "gs_2": 0,
            },
        ]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        pitcher_pred = [p for p in result.predictions if p.get("player_type") == "pitcher"][0]
        assert pitcher_pred["ip"] == 66.5

    def test_predict_empty_data(self) -> None:
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        assert result.model_name == "playing_time"
        assert len(result.predictions) == 0

    def test_predict_projected_season(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        assert result.predictions[0]["season"] == 2024

    def test_predict_multiple_batters(self) -> None:
        batting_rows = [
            {"player_id": 1, "season": 2023, "age": 29, "pa_1": 600, "pa_2": 550},
            {"player_id": 2, "season": 2023, "age": 25, "pa_1": 500, "pa_2": 450},
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(seasons=[2023])
        result = PlayingTimeModel(assembler=assembler).predict(config)
        assert len(result.predictions) == 2
