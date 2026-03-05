import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.config_yahoo import load_yahoo_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain import (
    ConfigError,
    Err,
    LabelConfig,
    LabeledSeason,
    Ok,
    Result,
)
from fantasy_baseball_manager.features import SqliteDatasetAssembler
from fantasy_baseball_manager.ingest import (
    ChadwickRegisterSource,
    FgStatsSource,
    LahmanAppearancesSource,
    LahmanPeopleSource,
    LahmanTeamsSource,
    MLBMinorLeagueBattingSource,
    MLBTransactionsSource,
    SprintSpeedSource,
    StatcastSavantSource,
)
from fantasy_baseball_manager.models import RunManager, get
from fantasy_baseball_manager.models.composite.engine import resolve_engine
from fantasy_baseball_manager.repos import (
    ADPRepo,
    KeeperCostRepo,
    PlayerRepo,
    ProjectionRepo,
    SqliteADPRepo,
    SqliteBattingStatsRepo,
    SqliteCheckpointRepo,
    SqliteExperimentRepo,
    SqliteFeatureCandidateRepo,
    SqliteILStintRepo,
    SqliteKeeperCostRepo,
    SqliteLeagueEnvironmentRepo,
    SqliteLevelFactorRepo,
    SqliteLoadLogRepo,
    SqliteMinorLeagueBattingStatsRepo,
    SqliteModelRunRepo,
    SqlitePitchingStatsRepo,
    SqlitePlayerRepo,
    SqlitePositionAppearanceRepo,
    SqliteProjectionRepo,
    SqliteRosterStintRepo,
    SqliteSprintSpeedRepo,
    SqliteStatcastPitchRepo,
    SqliteTeamRepo,
    SqliteValuationRepo,
    SqliteYahooDraftRepo,
    SqliteYahooLeagueRepo,
    SqliteYahooPlayerMapRepo,
    SqliteYahooRosterRepo,
    SqliteYahooTeamRepo,
    SqliteYahooTransactionRepo,
    ValuationRepo,
    YahooDraftRepo,
    YahooLeagueRepo,
    YahooPlayerMapRepo,
    YahooRosterRepo,
    YahooTeamRepo,
    YahooTransactionRepo,
)
from fantasy_baseball_manager.services import (
    CorrelationScanner,
    DatasetCatalogService,
    InjuryProfiler,
    LeagueEnvironmentService,
    PlayerEligibilityService,
    ProjectionEvaluator,
    StatcastColumnProfiler,
    StatsBasedPlayerUniverse,
    TemporalStabilityChecker,
    generate_labels,
)
from fantasy_baseball_manager.yahoo.auth import YahooAuth
from fantasy_baseball_manager.yahoo.client import YahooFantasyClient

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator

    from fantasy_baseball_manager.ingest import DataSource
    from fantasy_baseball_manager.models import Model, ModelConfig
    from fantasy_baseball_manager.services import (
        ADPAccuracyEvaluator,
        ADPMoversService,
        ADPReportService,
        PerformanceReportService,
        PlayerBiographyService,
        PlayerProfileService,
        ProjectionLookupService,
        ResidualAnalysisDiagnostic,
        ResidualAnalyzer,
        ResidualPersistenceDiagnostic,
        TrueTalentEvaluator,
        ValuationEvaluator,
        ValuationLookupService,
    )
    from fantasy_baseball_manager.team_resolver import TeamResolver


class DbLabelSource:
    """LabelSource that generates labels from ADP and valuation data in the database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._adp_repo = SqliteADPRepo(conn)
        self._val_repo = SqliteValuationRepo(conn)

    def get_labels(self, season: int) -> list[LabeledSeason]:
        adp = self._adp_repo.get_by_season(season)
        all_vals = self._val_repo.get_by_season(season, system="z")
        vals = [v for v in all_vals if v.projection_system == "actual"]
        return generate_labels(adp, vals, LabelConfig())


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
        engine = resolve_engine(config.model_params)
        position_appearance_repo = SqlitePositionAppearanceRepo(conn)
        batting_stats_repo = SqliteBattingStatsRepo(conn)
        pitching_stats_repo = SqlitePitchingStatsRepo(conn)
        eligibility_service = PlayerEligibilityService(
            position_appearance_repo,
            pitching_stats_repo=pitching_stats_repo,
        )
        player_universe = StatsBasedPlayerUniverse(
            batting_repo=batting_stats_repo,
            pitching_repo=pitching_stats_repo,
        )
        label_source = DbLabelSource(conn)
        result = create_model(
            model_name,
            assembler=assembler,
            engine=engine,
            projection_repo=SqliteProjectionRepo(conn),
            evaluator=evaluator,
            milb_repo=SqliteMinorLeagueBattingStatsRepo(conn),
            league_env_repo=SqliteLeagueEnvironmentRepo(conn),
            level_factor_repo=SqliteLevelFactorRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
            position_repo=position_appearance_repo,
            valuation_repo=SqliteValuationRepo(conn),
            eligibility_service=eligibility_service,
            player_universe=player_universe,
            label_source=label_source,
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
        container = AnalysisContainer(conn)
        yield EvalContext(
            conn=conn,
            evaluator=container.projection_evaluator,
            player_repo=container.player_repo,
            batting_repo=container.batting_stats_repo,
            projection_repo=container.projection_repo,
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
class ExperimentContext:
    conn: sqlite3.Connection
    repo: SqliteExperimentRepo
    checkpoint_repo: SqliteCheckpointRepo


@contextmanager
def build_experiment_context(data_dir: str) -> Iterator[ExperimentContext]:
    """Composition-root context manager for experiment subcommands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield ExperimentContext(
            conn=conn,
            repo=SqliteExperimentRepo(conn),
            checkpoint_repo=SqliteCheckpointRepo(conn),
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class DatasetsContext:
    conn: sqlite3.Connection
    catalog: DatasetCatalogService


@contextmanager
def build_datasets_context(data_dir: str) -> Iterator[DatasetsContext]:  # pragma: no cover
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
        assert self._statcast_conn is not None, "statcast_conn not configured"  # noqa: S101 - type narrowing
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
        return FgStatsSource(stat_type="bat")

    def pitching_source(self) -> DataSource:
        return FgStatsSource(stat_type="pit")

    @functools.cached_property
    def il_stint_repo(self) -> SqliteILStintRepo:  # pragma: no cover
        return SqliteILStintRepo(self._conn)

    def il_source(self) -> DataSource:  # pragma: no cover
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
    def sprint_speed_repo(self) -> SqliteSprintSpeedRepo:  # pragma: no cover
        return SqliteSprintSpeedRepo(self.statcast_conn)

    def sprint_speed_source(self) -> DataSource:
        return SprintSpeedSource()

    def milb_batting_source(self) -> DataSource:  # pragma: no cover
        return MLBMinorLeagueBattingSource()

    def appearances_source(self) -> DataSource:
        return LahmanAppearancesSource()

    def teams_source(self) -> DataSource:
        return LahmanTeamsSource()

    @functools.cached_property
    def adp_repo(self) -> SqliteADPRepo:  # pragma: no cover
        return SqliteADPRepo(self._conn)


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
        container = AnalysisContainer(conn)
        yield ProjectionsContext(conn=conn, lookup_service=container.projection_lookup_service)
    finally:
        conn.close()


@dataclass(frozen=True)
class BioContext:
    bio_service: PlayerBiographyService
    team_resolver: TeamResolver
    team_repo: SqliteTeamRepo


@contextmanager
def build_bio_context(data_dir: str) -> Iterator[BioContext]:
    """Composition-root context manager for bio commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield BioContext(
            bio_service=container.player_bio_service,
            team_resolver=container.team_resolver,
            team_repo=container.team_repo,
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class ReportContext:
    conn: sqlite3.Connection
    report_service: PerformanceReportService
    talent_evaluator: TrueTalentEvaluator
    residual_diagnostic: ResidualPersistenceDiagnostic
    residual_analysis_diagnostic: ResidualAnalysisDiagnostic


@contextmanager
def build_report_context(data_dir: str) -> Iterator[ReportContext]:
    """Composition-root context manager for report commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield ReportContext(
            conn=conn,
            report_service=container.performance_report_service,
            talent_evaluator=container.talent_evaluator,
            residual_diagnostic=container.residual_diagnostic,
            residual_analysis_diagnostic=container.residual_analysis_diagnostic,
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class ResidualsContext:
    conn: sqlite3.Connection
    analyzer: ResidualAnalyzer


@contextmanager
def build_residuals_context(data_dir: str) -> Iterator[ResidualsContext]:
    """Composition-root context manager for residuals commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield ResidualsContext(conn=conn, analyzer=container.residual_analyzer)
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
    def level_factor_repo(self) -> SqliteLevelFactorRepo:  # pragma: no cover
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
def build_compute_container(data_dir: str) -> Iterator[ComputeContainer]:  # pragma: no cover
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
        container = AnalysisContainer(conn)
        yield ValuationsContext(conn=conn, lookup_service=container.valuation_lookup_service)
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
        container = AnalysisContainer(conn)
        yield ValuationEvalContext(conn=conn, evaluator=container.valuation_evaluator)
    finally:
        conn.close()


@dataclass(frozen=True)
class ADPReportContext:
    conn: sqlite3.Connection
    service: ADPReportService


@contextmanager
def build_adp_report_context(data_dir: str) -> Iterator[ADPReportContext]:  # pragma: no cover
    """Composition-root context manager for ADP report commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield ADPReportContext(conn=conn, service=container.adp_report_service)
    finally:
        conn.close()


@dataclass(frozen=True)
class DraftBoardContext:
    conn: sqlite3.Connection
    player_repo: SqlitePlayerRepo
    valuation_repo: SqliteValuationRepo
    adp_repo: SqliteADPRepo
    profile_service: PlayerProfileService
    projection_repo: SqliteProjectionRepo


@contextmanager
def build_draft_board_context(data_dir: str) -> Iterator[DraftBoardContext]:
    """Composition-root context manager for draft board commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield DraftBoardContext(
            conn=conn,
            player_repo=container.player_repo,
            valuation_repo=container.valuation_repo,
            adp_repo=container.adp_repo,
            profile_service=container.player_profile_service,
            projection_repo=container.projection_repo,
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class CategoryNeedsContext:
    conn: sqlite3.Connection
    player_repo: SqlitePlayerRepo
    projection_repo: SqliteProjectionRepo


@contextmanager
def build_category_needs_context(data_dir: str) -> Iterator[CategoryNeedsContext]:
    """Composition-root context manager for category needs commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield CategoryNeedsContext(
            conn=conn,
            player_repo=SqlitePlayerRepo(conn),
            projection_repo=SqliteProjectionRepo(conn),
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class ADPAccuracyContext:
    conn: sqlite3.Connection
    evaluator: ADPAccuracyEvaluator


@contextmanager
def build_adp_accuracy_context(data_dir: str) -> Iterator[ADPAccuracyContext]:  # pragma: no cover
    """Composition-root context manager for ADP accuracy evaluation commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield ADPAccuracyContext(conn=conn, evaluator=container.adp_accuracy_evaluator)
    finally:
        conn.close()


@dataclass(frozen=True)
class ADPMoversContext:
    conn: sqlite3.Connection
    service: ADPMoversService


@contextmanager
def build_adp_movers_context(data_dir: str) -> Iterator[ADPMoversContext]:  # pragma: no cover
    """Composition-root context manager for ADP movers commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield ADPMoversContext(conn=conn, service=container.adp_movers_service)
    finally:
        conn.close()


@dataclass(frozen=True)
class ConfidenceReportContext:
    conn: sqlite3.Connection
    player_repo: SqlitePlayerRepo
    projection_repo: SqliteProjectionRepo
    valuation_repo: SqliteValuationRepo
    adp_repo: SqliteADPRepo


@contextmanager
def build_confidence_report_context(data_dir: str) -> Iterator[ConfidenceReportContext]:  # pragma: no cover
    """Composition-root context manager for projection confidence report commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(conn)
        yield ConfidenceReportContext(
            conn=conn,
            player_repo=container.player_repo,
            projection_repo=container.projection_repo,
            valuation_repo=container.valuation_repo,
            adp_repo=container.adp_repo,
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class ChatContext:
    conn: sqlite3.Connection
    container: AnalysisContainer


@contextmanager
def build_chat_context(data_dir: str, *, check_same_thread: bool = True) -> Iterator[ChatContext]:
    """Composition-root context manager for the chat/discord commands."""
    conn = create_connection(Path(data_dir) / "fbm.db", check_same_thread=check_same_thread)
    try:
        container = AnalysisContainer(conn)
        yield ChatContext(conn=conn, container=container)
    finally:
        conn.close()


@dataclass(frozen=True)
class YahooContext:
    conn: sqlite3.Connection
    yahoo_league_repo: YahooLeagueRepo
    yahoo_team_repo: YahooTeamRepo
    yahoo_player_map_repo: YahooPlayerMapRepo
    yahoo_roster_repo: YahooRosterRepo
    yahoo_draft_repo: YahooDraftRepo
    yahoo_transaction_repo: YahooTransactionRepo
    player_repo: PlayerRepo
    projection_repo: ProjectionRepo
    valuation_repo: ValuationRepo
    adp_repo: ADPRepo
    keeper_repo: KeeperCostRepo
    client: YahooFantasyClient


@contextmanager
def build_yahoo_context(data_dir: str, config_dir: Path) -> Iterator[YahooContext]:  # pragma: no cover
    """Composition-root context manager for Yahoo Fantasy commands."""
    config = load_yahoo_config(config_dir)
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        auth = YahooAuth(config.client_id, config.client_secret)
        client = YahooFantasyClient(auth)
        yield YahooContext(
            conn=conn,
            yahoo_league_repo=SqliteYahooLeagueRepo(conn),
            yahoo_team_repo=SqliteYahooTeamRepo(conn),
            yahoo_player_map_repo=SqliteYahooPlayerMapRepo(conn),
            yahoo_roster_repo=SqliteYahooRosterRepo(conn),
            yahoo_draft_repo=SqliteYahooDraftRepo(conn),
            yahoo_transaction_repo=SqliteYahooTransactionRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
            projection_repo=SqliteProjectionRepo(conn),
            valuation_repo=SqliteValuationRepo(conn),
            adp_repo=SqliteADPRepo(conn),
            keeper_repo=SqliteKeeperCostRepo(conn),
            client=client,
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class KeeperContext:
    conn: sqlite3.Connection
    keeper_repo: SqliteKeeperCostRepo
    player_repo: SqlitePlayerRepo
    valuation_repo: SqliteValuationRepo
    projection_repo: SqliteProjectionRepo
    eligibility_service: PlayerEligibilityService
    adp_repo: SqliteADPRepo


@contextmanager
def build_keeper_context(data_dir: str) -> Iterator[KeeperContext]:  # pragma: no cover
    """Composition-root context manager for keeper commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        position_repo = SqlitePositionAppearanceRepo(conn)
        pitching_stats_repo = SqlitePitchingStatsRepo(conn)
        eligibility_service = PlayerEligibilityService(
            position_repo,
            pitching_stats_repo=pitching_stats_repo,
        )
        yield KeeperContext(
            conn=conn,
            keeper_repo=SqliteKeeperCostRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
            valuation_repo=SqliteValuationRepo(conn),
            projection_repo=SqliteProjectionRepo(conn),
            eligibility_service=eligibility_service,
            adp_repo=SqliteADPRepo(conn),
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class ProfileContext:
    profiler: StatcastColumnProfiler
    scanner: CorrelationScanner
    stability_checker: TemporalStabilityChecker


@contextmanager
def build_profile_context(data_dir: str) -> Iterator[ProfileContext]:  # pragma: no cover
    """Composition-root context manager for profile commands."""
    statcast_conn = create_statcast_connection(Path(data_dir) / "statcast.db")
    stats_conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        scanner = CorrelationScanner(statcast_conn, stats_conn)
        yield ProfileContext(
            profiler=StatcastColumnProfiler(statcast_conn),
            scanner=scanner,
            stability_checker=TemporalStabilityChecker(scanner),
        )
    finally:
        stats_conn.close()
        statcast_conn.close()


@dataclass(frozen=True)
class FeatureContext:
    statcast_conn: sqlite3.Connection
    fbm_conn: sqlite3.Connection
    candidate_repo: SqliteFeatureCandidateRepo
    scanner: CorrelationScanner


@contextmanager
def build_feature_context(data_dir: str) -> Iterator[FeatureContext]:
    """Composition-root context manager for feature candidate commands."""
    statcast_conn = create_statcast_connection(Path(data_dir) / "statcast.db")
    fbm_conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield FeatureContext(
            statcast_conn=statcast_conn,
            fbm_conn=fbm_conn,
            candidate_repo=SqliteFeatureCandidateRepo(fbm_conn),
            scanner=CorrelationScanner(statcast_conn, fbm_conn),
        )
    finally:
        fbm_conn.close()
        statcast_conn.close()


@dataclass(frozen=True)
class InjuryAdjustedValuationsContext:
    conn: sqlite3.Connection
    projection_repo: SqliteProjectionRepo
    player_repo: SqlitePlayerRepo
    valuation_repo: SqliteValuationRepo
    eligibility_service: PlayerEligibilityService
    profiler: InjuryProfiler


@contextmanager
def build_injury_adjusted_valuations_context(data_dir: str) -> Iterator[InjuryAdjustedValuationsContext]:
    """Composition-root context manager for injury-adjusted valuation commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        position_repo = SqlitePositionAppearanceRepo(conn)
        pitching_stats_repo = SqlitePitchingStatsRepo(conn)
        eligibility_service = PlayerEligibilityService(
            position_repo,
            pitching_stats_repo=pitching_stats_repo,
        )
        yield InjuryAdjustedValuationsContext(
            conn=conn,
            projection_repo=SqliteProjectionRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
            valuation_repo=SqliteValuationRepo(conn),
            eligibility_service=eligibility_service,
            profiler=InjuryProfiler(
                player_repo=SqlitePlayerRepo(conn),
                il_stint_repo=SqliteILStintRepo(conn),
            ),
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class InjuryProfileContext:
    conn: sqlite3.Connection
    profiler: InjuryProfiler


@contextmanager
def build_injury_profile_context(data_dir: str) -> Iterator[InjuryProfileContext]:
    """Composition-root context manager for injury profile commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield InjuryProfileContext(
            conn=conn,
            profiler=InjuryProfiler(
                player_repo=SqlitePlayerRepo(conn),
                il_stint_repo=SqliteILStintRepo(conn),
            ),
        )
    finally:
        conn.close()


@dataclass(frozen=True)
class BreakoutBustReportContext:
    conn: sqlite3.Connection
    model: Model


@contextmanager
def build_breakout_bust_report_context(data_dir: str) -> Iterator[BreakoutBustReportContext]:
    """Composition-root context manager for breakout/bust report commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        assembler = SqliteDatasetAssembler(conn, statcast_path=Path(data_dir) / "statcast.db")
        label_source = DbLabelSource(conn)
        result = create_model("breakout-bust", assembler=assembler, label_source=label_source)
        if isinstance(result, Err):
            raise RuntimeError(result.error.message)
        yield BreakoutBustReportContext(conn=conn, model=result.value)
    finally:
        conn.close()
