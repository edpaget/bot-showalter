import sqlite3

from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    ModelConfig,
    Predictable,
    Preparable,
    ProjectionModel,
    Trainable,
)


class TestMarcelModel:
    def test_is_projection_model(self) -> None:
        assert isinstance(MarcelModel(), ProjectionModel)

    def test_is_preparable(self) -> None:
        assert isinstance(MarcelModel(), Preparable)

    def test_is_trainable(self) -> None:
        assert isinstance(MarcelModel(), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(MarcelModel(), Evaluable)

    def test_is_not_predictable(self) -> None:
        assert not isinstance(MarcelModel(), Predictable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(MarcelModel(), FineTunable)

    def test_artifact_type(self) -> None:
        assert MarcelModel().artifact_type == "none"

    def test_name(self) -> None:
        assert MarcelModel().name == "marcel"

    def test_supported_operations(self) -> None:
        ops = MarcelModel().supported_operations
        assert ops == frozenset({"prepare", "train", "evaluate"})

    def test_train_returns_result(self) -> None:
        result = MarcelModel().train(ModelConfig())
        assert result.model_name == "marcel"
        assert isinstance(result.metrics, dict)

    def test_evaluate_returns_result(self) -> None:
        result = MarcelModel().evaluate(ModelConfig())
        assert result.model_name == "marcel"
        assert isinstance(result.metrics, dict)


class TestMarcelPrepare:
    def test_prepare_returns_result_with_row_count(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2023])
        result = MarcelModel().prepare(config, assembler)
        assert result.model_name == "marcel"
        assert result.rows_processed > 0

    def test_prepare_uses_feature_dsl(self, seeded_conn: sqlite3.Connection) -> None:
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2022, 2023])
        result = MarcelModel().prepare(config, assembler)
        assert result.rows_processed > 0
        assert result.artifacts_path == config.artifacts_dir

    def test_prepare_idempotent(self, seeded_conn: sqlite3.Connection) -> None:
        """Calling prepare twice with same config returns cached result."""
        assembler = SqliteDatasetAssembler(seeded_conn)
        config = ModelConfig(seasons=[2023])
        result1 = MarcelModel().prepare(config, assembler)
        result2 = MarcelModel().prepare(config, assembler)
        assert result1.rows_processed == result2.rows_processed
