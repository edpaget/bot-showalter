import pytest

from fantasy_baseball_manager.cli._dispatcher import dispatch, UnsupportedOperation
from fantasy_baseball_manager.models.protocols import (
    EvalResult,
    ModelConfig,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import _clear, register


class _FakePreparableOnly:
    @property
    def name(self) -> str:
        return "fake"

    @property
    def category(self) -> str:
        return "batting"

    @property
    def description(self) -> str:
        return "Fake model"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare"})

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name="fake", rows_processed=42, artifacts_path="/tmp")


class _FakeFullModel(_FakePreparableOnly):
    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "evaluate"})

    def train(self, config: ModelConfig) -> TrainResult:
        return TrainResult(model_name="fake", metrics={"rmse": 0.5}, artifacts_path="/tmp")

    def evaluate(self, config: ModelConfig) -> EvalResult:
        return EvalResult(model_name="fake", metrics={"mae": 0.3})


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()


class TestDispatch:
    def test_dispatch_prepare(self) -> None:
        register("fake")(_FakePreparableOnly)
        result = dispatch("prepare", "fake", ModelConfig())
        assert isinstance(result, PrepareResult)
        assert result.rows_processed == 42

    def test_dispatch_train(self) -> None:
        register("fake")(_FakeFullModel)
        result = dispatch("train", "fake", ModelConfig())
        assert isinstance(result, TrainResult)
        assert result.metrics == {"rmse": 0.5}

    def test_dispatch_evaluate(self) -> None:
        register("fake")(_FakeFullModel)
        result = dispatch("evaluate", "fake", ModelConfig())
        assert isinstance(result, EvalResult)

    def test_unsupported_operation_raises(self) -> None:
        register("fake")(_FakePreparableOnly)
        with pytest.raises(UnsupportedOperation, match="does not support 'train'"):
            dispatch("train", "fake", ModelConfig())

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="no model registered"):
            dispatch("train", "nonexistent", ModelConfig())

    def test_unknown_operation_raises(self) -> None:
        register("fake")(_FakePreparableOnly)
        with pytest.raises(UnsupportedOperation, match="does not support 'bogus'"):
            dispatch("bogus", "fake", ModelConfig())
