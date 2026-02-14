import pytest

from fantasy_baseball_manager.models.protocols import ModelConfig, PrepareResult
from fantasy_baseball_manager.models.registry import get, list_models, register, _clear


class _DummyModel:
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def category(self) -> str:
        return "batting"

    @property
    def description(self) -> str:
        return "A dummy model for testing"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare"})

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name="dummy", rows_processed=0, artifacts_path="")


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()


class TestRegister:
    def test_register_and_get(self) -> None:
        register("dummy")(_DummyModel)
        model = get("dummy")
        assert model.name == "dummy"

    def test_register_returns_class_unchanged(self) -> None:
        result = register("dummy")(_DummyModel)
        assert result is _DummyModel

    def test_duplicate_registration_raises(self) -> None:
        register("dummy")(_DummyModel)
        with pytest.raises(ValueError, match="already registered"):
            register("dummy")(_DummyModel)

    def test_get_missing_model_raises(self) -> None:
        with pytest.raises(KeyError, match="no model registered"):
            get("nonexistent")

    def test_get_instantiates_lazily(self) -> None:
        call_count = 0

        class _Tracked(_DummyModel):
            def __init__(self) -> None:
                nonlocal call_count
                call_count += 1

        register("tracked")(_Tracked)
        assert call_count == 0
        get("tracked")
        assert call_count == 1


class TestListModels:
    def test_list_empty(self) -> None:
        assert list_models() == []

    def test_list_returns_sorted_names(self) -> None:
        register("bravo")(_DummyModel)
        register("alpha")(_DummyModel)
        assert list_models() == ["alpha", "bravo"]
