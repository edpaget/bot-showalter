from fantasy_baseball_manager.models.batting.marcel import MarcelModel
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

    def test_name(self) -> None:
        assert MarcelModel().name == "marcel"

    def test_category(self) -> None:
        assert MarcelModel().category == "batting"

    def test_supported_operations(self) -> None:
        ops = MarcelModel().supported_operations
        assert ops == frozenset({"prepare", "train", "evaluate"})

    def test_prepare_returns_result(self) -> None:
        result = MarcelModel().prepare(ModelConfig())
        assert result.model_name == "marcel"
        assert result.rows_processed == 0

    def test_train_returns_result(self) -> None:
        result = MarcelModel().train(ModelConfig())
        assert result.model_name == "marcel"
        assert isinstance(result.metrics, dict)

    def test_evaluate_returns_result(self) -> None:
        result = MarcelModel().evaluate(ModelConfig())
        assert result.model_name == "marcel"
        assert isinstance(result.metrics, dict)
