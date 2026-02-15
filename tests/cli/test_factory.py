from sqlite3 import ProgrammingError

import pytest

from fantasy_baseball_manager.cli._dispatcher import dispatch
from fantasy_baseball_manager.cli.factory import build_model_context, create_model
from fantasy_baseball_manager.db.connection import create_connection
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


class TestBuildModelContext:
    def test_yields_model_context_with_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        register("witharg")(_WithArgModel)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        config = ModelConfig()
        with build_model_context("witharg", config) as ctx:
            assert isinstance(ctx.model, _WithArgModel)
            assert ctx.model.assembler is not None

    def test_run_manager_is_none_without_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        register("witharg")(_WithArgModel)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        config = ModelConfig()
        with build_model_context("witharg", config) as ctx:
            assert ctx.run_manager is None

    def test_run_manager_is_set_with_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        register("witharg")(_WithArgModel)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        config = ModelConfig(version="v1")
        with build_model_context("witharg", config) as ctx:
            assert ctx.run_manager is not None

    def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        register("witharg")(_WithArgModel)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        config = ModelConfig()
        with build_model_context("witharg", config) as ctx:
            conn = ctx.conn
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")

    def test_new_model_with_custom_dependency_works_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Register → build_model_context → dispatch with a custom dep requires
        zero changes to dispatcher or protocols."""

        class _CustomModel:
            def __init__(self, assembler: object = None, scorer: object = None) -> None:
                self.assembler = assembler
                self.scorer = scorer

            @property
            def name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "Model with custom dependency"

            @property
            def supported_operations(self) -> frozenset[str]:
                return frozenset({"prepare"})

            def prepare(self, config: ModelConfig) -> PrepareResult:
                return PrepareResult(model_name="custom", rows_processed=42, artifacts_path="")

        register("custom")(_CustomModel)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        config = ModelConfig()
        with build_model_context("custom", config) as ctx:
            result = dispatch("prepare", ctx.model, config, ctx.run_manager)

        assert isinstance(result, PrepareResult)
        assert result.model_name == "custom"
        assert result.rows_processed == 42

    def test_connection_closed_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        register("witharg")(_WithArgModel)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        config = ModelConfig()
        with pytest.raises(RuntimeError):
            with build_model_context("witharg", config) as ctx:
                conn = ctx.conn
                raise RuntimeError("boom")
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")
