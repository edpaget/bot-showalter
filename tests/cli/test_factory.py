from sqlite3 import ProgrammingError

import pytest

from fantasy_baseball_manager.cli._dispatcher import dispatch
from fantasy_baseball_manager.cli.factory import (
    IngestContainer,
    build_eval_context,
    build_import_context,
    build_ingest_container,
    build_model_context,
    build_runs_context,
    build_valuation_eval_context,
    build_valuations_context,
    create_model,
)
from fantasy_baseball_manager.domain.errors import ConfigError
from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.models.protocols import ModelConfig, PrepareResult
from fantasy_baseball_manager.models.registry import _clear, register
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.repos.statcast_pitch_repo import SqliteStatcastPitchRepo
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator
from fantasy_baseball_manager.services.valuation_evaluator import ValuationEvaluator
from fantasy_baseball_manager.services.valuation_lookup import ValuationLookupService


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


class _WithModelNameModel:
    def __init__(self, model_name: str = "default") -> None:
        self.model_name_value = model_name

    @property
    def name(self) -> str:
        return self.model_name_value

    @property
    def description(self) -> str:
        return "Model that accepts model_name"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare"})

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name=self.model_name_value, rows_processed=0, artifacts_path="")


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()


class TestCreateModel:
    def test_create_model_with_matching_kwargs(self) -> None:
        register("witharg")(_WithArgModel)
        sentinel = object()
        result = create_model("witharg", assembler=sentinel)
        assert isinstance(result, Ok)
        assert isinstance(result.value, _WithArgModel)
        assert result.value.assembler is sentinel

    def test_create_model_filters_non_matching_kwargs(self) -> None:
        register("noarg")(_NoArgModel)
        result = create_model("noarg", assembler=object(), extra=42)
        assert isinstance(result, Ok)
        assert isinstance(result.value, _NoArgModel)

    def test_create_model_passes_model_name(self) -> None:
        register("custom-variant")(_WithModelNameModel)
        result = create_model("custom-variant")
        assert isinstance(result, Ok)
        assert isinstance(result.value, _WithModelNameModel)
        assert result.value.model_name_value == "custom-variant"

    def test_create_model_does_not_pass_model_name_when_not_accepted(self) -> None:
        register("noarg")(_NoArgModel)
        result = create_model("noarg")
        assert isinstance(result, Ok)
        assert isinstance(result.value, _NoArgModel)

    def test_create_model_unknown_name_returns_err(self) -> None:
        result = create_model("nonexistent")
        assert isinstance(result, Err)
        assert isinstance(result.error, ConfigError)
        assert "nonexistent" in result.error.message


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

        assert isinstance(result, Ok)
        assert isinstance(result.value, PrepareResult)
        assert result.value.model_name == "custom"
        assert result.value.rows_processed == 42

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


class TestBuildEvalContext:
    def test_yields_eval_context_with_evaluator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_eval_context("./data") as ctx:
            assert isinstance(ctx.evaluator, ProjectionEvaluator)

    def test_eval_context_has_projection_repo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_eval_context("./data") as ctx:
            assert isinstance(ctx.projection_repo, SqliteProjectionRepo)

    def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_eval_context("./data") as ctx:
            conn = ctx.conn
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")


class TestBuildRunsContext:
    def test_yields_runs_context_with_repo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_runs_context("./data") as ctx:
            assert isinstance(ctx.repo, SqliteModelRunRepo)

    def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_runs_context("./data") as ctx:
            conn = ctx.conn
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")


class TestBuildImportContext:
    def test_yields_import_context_with_repos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_import_context("./data") as ctx:
            assert isinstance(ctx.player_repo, SqlitePlayerRepo)
            assert isinstance(ctx.proj_repo, SqliteProjectionRepo)
            assert isinstance(ctx.log_repo, SqliteLoadLogRepo)

    def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_import_context("./data") as ctx:
            conn = ctx.conn
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")


class TestIngestContainer:
    def test_provides_repos(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        assert isinstance(container.player_repo, SqlitePlayerRepo)
        assert isinstance(container.batting_stats_repo, SqliteBattingStatsRepo)
        assert isinstance(container.pitching_stats_repo, SqlitePitchingStatsRepo)
        assert isinstance(container.log_repo, SqliteLoadLogRepo)
        conn.close()

    def test_conn_property(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        assert container.conn is conn
        conn.close()

    def test_batting_source_fangraphs(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.batting_source()
        assert source.source_type == "fangraphs"
        assert source.source_detail == "batting"
        conn.close()

    def test_pitching_source_fangraphs(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.pitching_source()
        assert source.source_type == "fangraphs"
        assert source.source_detail == "pitching"
        conn.close()

    def test_player_source(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.player_source()
        assert source.source_type == "chadwick_bureau"
        assert source.source_detail == "chadwick_register"
        conn.close()

    def test_statcast_pitch_repo(self) -> None:
        conn = create_connection(":memory:")
        statcast_conn = create_statcast_connection(":memory:")
        container = IngestContainer(conn, statcast_conn=statcast_conn)
        assert isinstance(container.statcast_pitch_repo, SqliteStatcastPitchRepo)
        statcast_conn.close()
        conn.close()

    def test_statcast_source(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.statcast_source()
        assert source.source_type == "baseball_savant"
        assert source.source_detail == "statcast_pitch"
        conn.close()

    def test_bio_source(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.bio_source()
        assert source.source_type == "lahman"
        assert source.source_detail == "people"
        conn.close()

    def test_appearances_source(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.appearances_source()
        assert source.source_type == "lahman"
        assert source.source_detail == "appearances"
        conn.close()

    def test_teams_source(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.teams_source()
        assert source.source_type == "lahman"
        assert source.source_detail == "teams"
        conn.close()

    def test_sprint_speed_source(self) -> None:
        conn = create_connection(":memory:")
        container = IngestContainer(conn)
        source = container.sprint_speed_source()
        assert source.source_type == "baseball_savant"
        assert source.source_detail == "sprint_speed"
        conn.close()

    def test_statcast_conn_property(self) -> None:
        conn = create_connection(":memory:")
        statcast_conn = create_statcast_connection(":memory:")
        container = IngestContainer(conn, statcast_conn=statcast_conn)
        assert container.statcast_conn is statcast_conn
        statcast_conn.close()
        conn.close()


class TestBuildIngestContainer:
    def test_yields_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_statcast_connection",
            lambda path: create_statcast_connection(":memory:"),
        )
        with build_ingest_container("./data") as container:
            assert isinstance(container, IngestContainer)
            assert container.statcast_conn is not None

    def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_statcast_connection",
            lambda path: create_statcast_connection(":memory:"),
        )
        with build_ingest_container("./data") as container:
            conn = container.conn
            statcast_conn = container.statcast_conn
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")
        with pytest.raises(ProgrammingError):
            statcast_conn.execute("SELECT 1")


class TestBuildValuationsContext:
    def test_yields_valuations_context_with_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_valuations_context("./data") as ctx:
            assert isinstance(ctx.lookup_service, ValuationLookupService)

    def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_valuations_context("./data") as ctx:
            conn = ctx.conn
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")


class TestBuildValuationEvalContext:
    def test_yields_valuation_eval_context_with_evaluator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_valuation_eval_context("./data") as ctx:
            assert isinstance(ctx.evaluator, ValuationEvaluator)

    def test_connection_closed_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(":memory:"),
        )
        with build_valuation_eval_context("./data") as ctx:
            conn = ctx.conn
        with pytest.raises(ProgrammingError):
            conn.execute("SELECT 1")
