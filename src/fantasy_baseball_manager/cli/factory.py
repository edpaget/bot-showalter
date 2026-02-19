import functools
import inspect
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.ingest.mlb_milb_stats_source import MLBMinorLeagueBattingSource
from fantasy_baseball_manager.ingest.mlb_transactions_source import MLBTransactionsSource
from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.chadwick_source import ChadwickRegisterSource
from fantasy_baseball_manager.ingest.lahman_source import LahmanAppearancesSource, LahmanPeopleSource, LahmanTeamsSource
from fantasy_baseball_manager.ingest.fangraphs_source import FgBattingSource, FgPitchingSource
from fantasy_baseball_manager.ingest.sprint_speed_source import SprintSpeedSource
from fantasy_baseball_manager.ingest.statcast_savant_source import StatcastSavantSource
from fantasy_baseball_manager.domain.errors import ConfigError
from fantasy_baseball_manager.domain.result import Err, Ok, Result
from fantasy_baseball_manager.models.protocols import Model, ModelConfig
from fantasy_baseball_manager.models.registry import get
from fantasy_baseball_manager.models.run_manager import RunManager
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.il_stint_repo import SqliteILStintRepo
from fantasy_baseball_manager.repos.league_environment_repo import SqliteLeagueEnvironmentRepo
from fantasy_baseball_manager.repos.level_factor_repo import SqliteLevelFactorRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.minor_league_batting_stats_repo import SqliteMinorLeagueBattingStatsRepo
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from fantasy_baseball_manager.repos.sprint_speed_repo import SqliteSprintSpeedRepo
from fantasy_baseball_manager.repos.statcast_pitch_repo import SqliteStatcastPitchRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.services.dataset_catalog import DatasetCatalogService
from fantasy_baseball_manager.services.league_environment_service import LeagueEnvironmentService
from fantasy_baseball_manager.services.performance_report import PerformanceReportService
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator
from fantasy_baseball_manager.services.residual_persistence_diagnostic import ResidualPersistenceDiagnostic
from fantasy_baseball_manager.services.true_talent_evaluator import TrueTalentEvaluator
from fantasy_baseball_manager.services.projection_lookup import ProjectionLookupService
from fantasy_baseball_manager.services.valuation_evaluator import ValuationEvaluator
from fantasy_baseball_manager.services.valuation_lookup import ValuationLookupService


def create_model(name: str, **kwargs: Any) -> Result[Model, ConfigError]:
    """Look up a model class by name and instantiate it, forwarding matching kwargs.

    When called without all required deps (e.g. for ``info`` or ``features``
    commands), missing required parameters are filled with ``None`` so that
    metadata properties (name, description, supported_operations) still work.
    """
    try:
        cls = get(name)
    except KeyError:
        return Err(ConfigError(message=f"'{name}': no model registered with this name"))
    sig = inspect.signature(cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if "model_name" in sig.parameters:
        filtered["model_name"] = name
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param_name not in filtered and param.default is inspect.Parameter.empty:
            filtered[param_name] = None
    return Ok(cls(**filtered))


@dataclass(frozen=True)
class ModelContext:
    conn: sqlite3.Connection
    model: Model
    run_manager: RunManager | None
    projection_repo: SqliteProjectionRepo | None = None


@contextmanager
def build_model_context(model_name: str, config: ModelConfig) -> Iterator[ModelContext]:
    """Composition-root context manager: opens DB, wires assembler + model, yields context, closes DB."""
    conn = create_connection(Path(config.data_dir) / "fbm.db")
    try:
        assembler = SqliteDatasetAssembler(conn, statcast_path=Path(config.data_dir) / "statcast.db")
        evaluator = ProjectionEvaluator(
            SqliteProjectionRepo(conn),
            SqliteBattingStatsRepo(conn),
            SqlitePitchingStatsRepo(conn),
        )
        result = create_model(
            model_name,
            assembler=assembler,
            projection_repo=SqliteProjectionRepo(conn),
            evaluator=evaluator,
            milb_repo=SqliteMinorLeagueBattingStatsRepo(conn),
            league_env_repo=SqliteLeagueEnvironmentRepo(conn),
            level_factor_repo=SqliteLevelFactorRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
            position_repo=SqlitePositionAppearanceRepo(conn),
            valuation_repo=SqliteValuationRepo(conn),
        )
        if isinstance(result, Err):
            raise RuntimeError(result.error.message)
        model = result.value

        run_manager: RunManager | None = None
        if config.version is not None:
            repo = SqliteModelRunRepo(conn)
            run_manager = RunManager(model_run_repo=repo, artifacts_root=Path(config.artifacts_dir))

        projection_repo = SqliteProjectionRepo(conn)
        yield ModelContext(conn=conn, model=model, run_manager=run_manager, projection_repo=projection_repo)
    finally:
        conn.close()


@dataclass(frozen=True)
class EvalContext:
    conn: sqlite3.Connection
    evaluator: ProjectionEvaluator
    player_repo: SqlitePlayerRepo
    batting_repo: SqliteBattingStatsRepo
    projection_repo: SqliteProjectionRepo


@contextmanager
def build_eval_context(data_dir: str) -> Iterator[EvalContext]:
    """Composition-root context manager for eval/compare commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        projection_repo = SqliteProjectionRepo(conn)
        batting_repo = SqliteBattingStatsRepo(conn)
        evaluator = ProjectionEvaluator(
            projection_repo,
            batting_repo,
            SqlitePitchingStatsRepo(conn),
        )
        yield EvalContext(
            conn=conn,
            evaluator=evaluator,
            player_repo=SqlitePlayerRepo(conn),
            batting_repo=batting_repo,
            projection_repo=projection_repo,
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class RunsContext:
    conn: sqlite3.Connection
    repo: SqliteModelRunRepo


@contextmanager
def build_runs_context(data_dir: str) -> Iterator[RunsContext]:
    """Composition-root context manager for runs subcommands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield RunsContext(conn=conn, repo=SqliteModelRunRepo(conn))
    finally:
        conn.close()


@dataclass(frozen=True)
class DatasetsContext:
    conn: sqlite3.Connection
    catalog: DatasetCatalogService


@contextmanager
def build_datasets_context(data_dir: str) -> Iterator[DatasetsContext]:
    """Composition-root context manager for datasets subcommands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield DatasetsContext(conn=conn, catalog=DatasetCatalogService(conn))
    finally:
        conn.close()


@dataclass(frozen=True)
class ImportContext:
    conn: sqlite3.Connection
    player_repo: SqlitePlayerRepo
    proj_repo: SqliteProjectionRepo
    log_repo: SqliteLoadLogRepo


@contextmanager
def build_import_context(data_dir: str) -> Iterator[ImportContext]:
    """Composition-root context manager for the import command."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield ImportContext(
            conn=conn,
            player_repo=SqlitePlayerRepo(conn),
            proj_repo=SqliteProjectionRepo(conn),
            log_repo=SqliteLoadLogRepo(conn),
        )
    finally:
        conn.close()


class IngestContainer:
    """DI container for the ingest command group."""

    def __init__(self, conn: sqlite3.Connection, *, statcast_conn: sqlite3.Connection | None = None) -> None:
        self._conn = conn
        self._statcast_conn = statcast_conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def statcast_conn(self) -> sqlite3.Connection:
        assert self._statcast_conn is not None, "statcast_conn not configured"
        return self._statcast_conn

    @functools.cached_property
    def statcast_pitch_repo(self) -> SqliteStatcastPitchRepo:
        return SqliteStatcastPitchRepo(self.statcast_conn)

    def statcast_source(self) -> DataSource:
        return StatcastSavantSource()

    @functools.cached_property
    def player_repo(self) -> SqlitePlayerRepo:
        return SqlitePlayerRepo(self._conn)

    @functools.cached_property
    def batting_stats_repo(self) -> SqliteBattingStatsRepo:
        return SqliteBattingStatsRepo(self._conn)

    @functools.cached_property
    def pitching_stats_repo(self) -> SqlitePitchingStatsRepo:
        return SqlitePitchingStatsRepo(self._conn)

    @functools.cached_property
    def log_repo(self) -> SqliteLoadLogRepo:
        return SqliteLoadLogRepo(self._conn)

    def player_source(self) -> DataSource:
        return ChadwickRegisterSource()

    def batting_source(self) -> DataSource:
        return FgBattingSource()

    def pitching_source(self) -> DataSource:
        return FgPitchingSource()

    @functools.cached_property
    def il_stint_repo(self) -> SqliteILStintRepo:
        return SqliteILStintRepo(self._conn)

    def il_source(self) -> DataSource:
        return MLBTransactionsSource()

    def bio_source(self) -> DataSource:
        return LahmanPeopleSource()

    @functools.cached_property
    def position_appearance_repo(self) -> SqlitePositionAppearanceRepo:
        return SqlitePositionAppearanceRepo(self._conn)

    @functools.cached_property
    def roster_stint_repo(self) -> SqliteRosterStintRepo:
        return SqliteRosterStintRepo(self._conn)

    @functools.cached_property
    def team_repo(self) -> SqliteTeamRepo:
        return SqliteTeamRepo(self._conn)

    @functools.cached_property
    def minor_league_batting_stats_repo(self) -> SqliteMinorLeagueBattingStatsRepo:
        return SqliteMinorLeagueBattingStatsRepo(self._conn)

    @functools.cached_property
    def sprint_speed_repo(self) -> SqliteSprintSpeedRepo:
        return SqliteSprintSpeedRepo(self.statcast_conn)

    def sprint_speed_source(self) -> DataSource:
        return SprintSpeedSource()

    def milb_batting_source(self) -> DataSource:
        return MLBMinorLeagueBattingSource()

    def appearances_source(self) -> DataSource:
        return LahmanAppearancesSource()

    def teams_source(self) -> DataSource:
        return LahmanTeamsSource()


@contextmanager
def build_ingest_container(data_dir: str) -> Iterator[IngestContainer]:
    """Composition-root context manager for ingest commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    statcast_conn = create_statcast_connection(Path(data_dir) / "statcast.db")
    try:
        yield IngestContainer(conn, statcast_conn=statcast_conn)
    finally:
        statcast_conn.close()
        conn.close()


@dataclass(frozen=True)
class ProjectionsContext:
    conn: sqlite3.Connection
    lookup_service: ProjectionLookupService


@contextmanager
def build_projections_context(data_dir: str) -> Iterator[ProjectionsContext]:
    """Composition-root context manager for projections commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        lookup_service = ProjectionLookupService(
            SqlitePlayerRepo(conn),
            SqliteProjectionRepo(conn),
        )
        yield ProjectionsContext(conn=conn, lookup_service=lookup_service)
    finally:
        conn.close()


@dataclass(frozen=True)
class ReportContext:
    conn: sqlite3.Connection
    report_service: PerformanceReportService
    talent_evaluator: TrueTalentEvaluator
    residual_diagnostic: ResidualPersistenceDiagnostic


@contextmanager
def build_report_context(data_dir: str) -> Iterator[ReportContext]:
    """Composition-root context manager for report commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        proj_repo = SqliteProjectionRepo(conn)
        batting_repo = SqliteBattingStatsRepo(conn)
        pitching_repo = SqlitePitchingStatsRepo(conn)
        player_repo = SqlitePlayerRepo(conn)
        report_service = PerformanceReportService(
            proj_repo,
            player_repo,
            batting_repo,
            pitching_repo,
        )
        talent_evaluator = TrueTalentEvaluator(proj_repo, batting_repo, pitching_repo)
        residual_diagnostic = ResidualPersistenceDiagnostic(proj_repo, batting_repo, player_repo)
        yield ReportContext(
            conn=conn,
            report_service=report_service,
            talent_evaluator=talent_evaluator,
            residual_diagnostic=residual_diagnostic,
        )
    finally:
        conn.close()


class ComputeContainer:
    """DI container for the compute command group."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @functools.cached_property
    def league_environment_repo(self) -> SqliteLeagueEnvironmentRepo:
        return SqliteLeagueEnvironmentRepo(self._conn)

    @functools.cached_property
    def level_factor_repo(self) -> SqliteLevelFactorRepo:
        return SqliteLevelFactorRepo(self._conn)

    @functools.cached_property
    def minor_league_batting_stats_repo(self) -> SqliteMinorLeagueBattingStatsRepo:
        return SqliteMinorLeagueBattingStatsRepo(self._conn)

    @functools.cached_property
    def league_environment_service(self) -> LeagueEnvironmentService:
        return LeagueEnvironmentService(
            self.minor_league_batting_stats_repo,
            self.league_environment_repo,
        )


@contextmanager
def build_compute_container(data_dir: str) -> Iterator[ComputeContainer]:
    """Composition-root context manager for compute commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield ComputeContainer(conn)
    finally:
        conn.close()


@dataclass(frozen=True)
class ValuationsContext:
    conn: sqlite3.Connection
    lookup_service: ValuationLookupService


@contextmanager
def build_valuations_context(data_dir: str) -> Iterator[ValuationsContext]:
    """Composition-root context manager for valuations commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        lookup_service = ValuationLookupService(
            SqlitePlayerRepo(conn),
            SqliteValuationRepo(conn),
        )
        yield ValuationsContext(conn=conn, lookup_service=lookup_service)
    finally:
        conn.close()


@dataclass(frozen=True)
class ValuationEvalContext:
    conn: sqlite3.Connection
    evaluator: ValuationEvaluator


@contextmanager
def build_valuation_eval_context(data_dir: str) -> Iterator[ValuationEvalContext]:
    """Composition-root context manager for valuation evaluation commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        evaluator = ValuationEvaluator(
            valuation_repo=SqliteValuationRepo(conn),
            batting_repo=SqliteBattingStatsRepo(conn),
            pitching_repo=SqlitePitchingStatsRepo(conn),
            position_repo=SqlitePositionAppearanceRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
        )
        yield ValuationEvalContext(conn=conn, evaluator=evaluator)
    finally:
        conn.close()
