import pytest

from fantasy_baseball_manager.models.registry import get, list_models, register, register_alias, _clear


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


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()


class TestRegister:
    def test_register_and_get_returns_class(self) -> None:
        register("dummy")(_DummyModel)
        cls = get("dummy")
        assert cls is _DummyModel

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


class TestRegisterAlias:
    def test_register_alias_resolves_to_same_class(self) -> None:
        register("foo")(_DummyModel)
        register_alias("bar", "foo")
        assert get("bar") is _DummyModel

    def test_register_alias_duplicate_name_raises(self) -> None:
        register("foo")(_DummyModel)
        with pytest.raises(ValueError, match="already registered"):
            register_alias("foo", "foo")

    def test_register_alias_missing_target_raises(self) -> None:
        with pytest.raises(KeyError, match="no model registered"):
            register_alias("bar", "nonexistent")

    def test_list_models_includes_aliases(self) -> None:
        register("foo")(_DummyModel)
        register_alias("bar", "foo")
        assert "bar" in list_models()
        assert "foo" in list_models()


class TestListModels:
    def test_list_empty(self) -> None:
        assert list_models() == []

    def test_list_returns_sorted_names(self) -> None:
        register("bravo")(_DummyModel)
        register("alpha")(_DummyModel)
        assert list_models() == ["alpha", "bravo"]
