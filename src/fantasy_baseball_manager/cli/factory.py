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
from fantasy_baseball_manager.ingest.pybaseball_source import (
    BrefBattingSource,
    BrefPitchingSource,
    ChadwickSource,
    FgBattingSource,
    FgPitchingSource,
    LahmanAppearancesSource,
    LahmanPeopleSource,
    LahmanTeamsSource,
    StatcastSource,
)
from fantasy_baseball_manager.models.protocols import Model, ModelConfig
from fantasy_baseball_manager.models.registry import get
from fantasy_baseball_manager.models.run_manager import RunManager
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.il_stint_repo import SqliteILStintRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.minor_league_batting_stats_repo import SqliteMinorLeagueBattingStatsRepo
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from fantasy_baseball_manager.repos.statcast_pitch_repo import SqliteStatcastPitchRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.performance_report import PerformanceReportService
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator
from fantasy_baseball_manager.services.projection_lookup import ProjectionLookupService


def create_model(name: str, **kwargs: Any) -> Model:
    """Look up a model class by name and instantiate it, forwarding matching kwargs."""
    cls = get(name)
    sig = inspect.signature(cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered)


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
        assembler = SqliteDatasetAssembler(conn)
        evaluator = ProjectionEvaluator(
            SqliteProjectionRepo(conn),
            SqliteBattingStatsRepo(conn),
            SqlitePitchingStatsRepo(conn),
        )
        model = create_model(
            model_name,
            assembler=assembler,
            projection_repo=SqliteProjectionRepo(conn),
            evaluator=evaluator,
        )

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


@contextmanager
def build_eval_context(data_dir: str) -> Iterator[EvalContext]:
    """Composition-root context manager for eval/compare commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        evaluator = ProjectionEvaluator(
            SqliteProjectionRepo(conn),
            SqliteBattingStatsRepo(conn),
            SqlitePitchingStatsRepo(conn),
        )
        yield EvalContext(conn=conn, evaluator=evaluator)
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

    @property
    def statcast_pitch_repo(self) -> SqliteStatcastPitchRepo:
        return SqliteStatcastPitchRepo(self.statcast_conn)

    def statcast_source(self) -> DataSource:
        return StatcastSource()

    @property
    def player_repo(self) -> SqlitePlayerRepo:
        return SqlitePlayerRepo(self._conn)

    @property
    def batting_stats_repo(self) -> SqliteBattingStatsRepo:
        return SqliteBattingStatsRepo(self._conn)

    @property
    def pitching_stats_repo(self) -> SqlitePitchingStatsRepo:
        return SqlitePitchingStatsRepo(self._conn)

    @property
    def log_repo(self) -> SqliteLoadLogRepo:
        return SqliteLoadLogRepo(self._conn)

    def player_source(self) -> DataSource:
        return ChadwickSource()

    def batting_source(self, name: str) -> DataSource:
        if name == "fangraphs":
            return FgBattingSource()
        if name == "bbref":
            return BrefBattingSource()
        raise ValueError(f"Unknown batting source: {name!r}")

    def pitching_source(self, name: str) -> DataSource:
        if name == "fangraphs":
            return FgPitchingSource()
        if name == "bbref":
            return BrefPitchingSource()
        raise ValueError(f"Unknown pitching source: {name!r}")

    @property
    def il_stint_repo(self) -> SqliteILStintRepo:
        return SqliteILStintRepo(self._conn)

    def il_source(self) -> DataSource:
        return MLBTransactionsSource()

    def bio_source(self) -> DataSource:
        return LahmanPeopleSource()

    @property
    def position_appearance_repo(self) -> SqlitePositionAppearanceRepo:
        return SqlitePositionAppearanceRepo(self._conn)

    @property
    def roster_stint_repo(self) -> SqliteRosterStintRepo:
        return SqliteRosterStintRepo(self._conn)

    @property
    def team_repo(self) -> SqliteTeamRepo:
        return SqliteTeamRepo(self._conn)

    @property
    def minor_league_batting_stats_repo(self) -> SqliteMinorLeagueBattingStatsRepo:
        return SqliteMinorLeagueBattingStatsRepo(self._conn)

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


@contextmanager
def build_report_context(data_dir: str) -> Iterator[ReportContext]:
    """Composition-root context manager for report commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        report_service = PerformanceReportService(
            SqliteProjectionRepo(conn),
            SqlitePlayerRepo(conn),
            SqliteBattingStatsRepo(conn),
            SqlitePitchingStatsRepo(conn),
        )
        yield ReportContext(conn=conn, report_service=report_service)
    finally:
        conn.close()
