import sqlite3
from typing import Any

from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    ModelConfig,
    Predictable,
    Preparable,
    Model,
    Trainable,
)


class TestMarcelModel:
    def test_is_model(self) -> None:
        assert isinstance(MarcelModel(), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(MarcelModel(), Preparable)

    def test_is_not_trainable(self) -> None:
        assert not isinstance(MarcelModel(), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(MarcelModel(), Evaluable)

    def test_is_predictable(self) -> None:
        assert isinstance(MarcelModel(), Predictable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(MarcelModel(), FineTunable)

    def test_artifact_type(self) -> None:
        assert MarcelModel().artifact_type == "none"

    def test_name(self) -> None:
        assert MarcelModel().name == "marcel"

    def test_supported_operations(self) -> None:
        ops = MarcelModel().supported_operations
        assert ops == frozenset({"prepare", "predict", "evaluate"})

    def test_evaluate_returns_result(self) -> None:
        result = MarcelModel().evaluate(ModelConfig())
        assert result.model_name == "marcel"
        assert isinstance(result.metrics, dict)


class TestMarcelPrepare:
    def test_prepare_returns_result_with_row_count(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2023])
        result = MarcelModel(assembler=assembler).prepare(config)
        assert result.model_name == "marcel"
        assert result.rows_processed > 0

    def test_prepare_uses_feature_dsl(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2022, 2023])
        result = MarcelModel(assembler=assembler).prepare(config)
        assert result.rows_processed > 0
        assert result.artifacts_path == config.artifacts_dir

    def test_prepare_idempotent(self, seeded_conn: sqlite3.Connection) -> None:
        """Calling prepare twice with same config returns cached result."""
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2023])
        model = MarcelModel(assembler=assembler)
        result1 = model.prepare(config)
        result2 = model.prepare(config)
        assert result1.rows_processed == result2.rows_processed


class FakeAssembler:
    """In-memory assembler for integration testing predict()."""

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


class TestMarcelPredict:
    def test_predict_returns_result(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
            {
                "player_id": 2,
                "season": 2023,
                "age": 25,
                "pa_1": 500,
                "pa_2": 450,
                "pa_3": 400,
                "hr_1": 20.0,
                "hr_2": 18.0,
                "hr_3": 15.0,
                "hr_wavg": 217.0 / 5500.0,
                "weighted_pt": 5500.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        model = MarcelModel(assembler=assembler)
        result = model.predict(config)
        assert result.model_name == "marcel"
        assert len(result.predictions) == 2

    def test_predict_with_pitchers(self) -> None:
        pitching_rows = [
            {
                "player_id": 10,
                "season": 2023,
                "age": 28,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "ip_3": 160.0,
                "g_1": 30,
                "g_2": 28,
                "g_3": 26,
                "gs_1": 30,
                "gs_2": 28,
                "gs_3": 26,
                "so_1": 200.0,
                "so_2": 180.0,
                "so_3": 150.0,
                "so_wavg": 1110.0 / 1040.0,
                "weighted_pt": 1040.0,
                "league_so_rate": 200.0 / 180.0,
            },
        ]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={
                "batting_categories": ["hr"],
                "pitching_categories": ["so"],
            },
        )
        model = MarcelModel(assembler=assembler)
        result = model.predict(config)
        assert result.model_name == "marcel"
        # Should have pitcher projections
        pitcher_preds = [p for p in result.predictions if p.get("player_type") == "pitcher"]
        assert len(pitcher_preds) == 1

    def test_predict_empty_data(self) -> None:
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        model = MarcelModel(assembler=assembler)
        result = model.predict(config)
        assert result.model_name == "marcel"
        assert len(result.predictions) == 0
