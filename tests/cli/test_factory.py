import pytest

from fantasy_baseball_manager.cli.factory import create_model
from fantasy_baseball_manager.models.protocols import ModelConfig, PrepareResult
from fantasy_baseball_manager.models.registry import _clear, register


class _NoArgModel:
    @property
    def name(self) -> str:
        return "noarg"

    @property
    def description(self) -> str:
        return "Model with no constructor args"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare"})

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name="noarg", rows_processed=0, artifacts_path="")


class _WithArgModel:
    def __init__(self, assembler: object = None) -> None:
        self.assembler = assembler

    @property
    def name(self) -> str:
        return "witharg"

    @property
    def description(self) -> str:
        return "Model that accepts assembler"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare"})

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name="witharg", rows_processed=0, artifacts_path="")


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()


class TestCreateModel:
    def test_create_model_with_matching_kwargs(self) -> None:
        register("witharg")(_WithArgModel)
        sentinel = object()
        model = create_model("witharg", assembler=sentinel)
        assert isinstance(model, _WithArgModel)
        assert model.assembler is sentinel

    def test_create_model_filters_non_matching_kwargs(self) -> None:
        register("noarg")(_NoArgModel)
        model = create_model("noarg", assembler=object(), extra=42)
        assert isinstance(model, _NoArgModel)

    def test_create_model_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="no model registered"):
            create_model("nonexistent")
