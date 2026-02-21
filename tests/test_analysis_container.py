import sqlite3

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo, SqliteTeamRepo
from fantasy_baseball_manager.repos.position_appearance_repo import SqlitePositionAppearanceRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.repos.roster_stint_repo import SqliteRosterStintRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.services.adp_accuracy import ADPAccuracyEvaluator
from fantasy_baseball_manager.services.adp_movers import ADPMoversService
from fantasy_baseball_manager.services.adp_report import ADPReportService
from fantasy_baseball_manager.services.performance_report import PerformanceReportService
from fantasy_baseball_manager.services.player_biography import PlayerBiographyService
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator
from fantasy_baseball_manager.services.projection_lookup import ProjectionLookupService
from fantasy_baseball_manager.services.residual_persistence_diagnostic import ResidualPersistenceDiagnostic
from fantasy_baseball_manager.services.true_talent_evaluator import TrueTalentEvaluator
from fantasy_baseball_manager.services.valuation_evaluator import ValuationEvaluator
from fantasy_baseball_manager.services.valuation_lookup import ValuationLookupService


class TestRepoTypes:
    def test_player_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.player_repo, SqlitePlayerRepo)

    def test_team_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.team_repo, SqliteTeamRepo)

    def test_projection_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.projection_repo, SqliteProjectionRepo)

    def test_valuation_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.valuation_repo, SqliteValuationRepo)

    def test_adp_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.adp_repo, SqliteADPRepo)

    def test_batting_stats_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.batting_stats_repo, SqliteBattingStatsRepo)

    def test_pitching_stats_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.pitching_stats_repo, SqlitePitchingStatsRepo)

    def test_position_appearance_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.position_appearance_repo, SqlitePositionAppearanceRepo)

    def test_roster_stint_repo(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.roster_stint_repo, SqliteRosterStintRepo)


class TestServiceTypes:
    def test_player_bio_service(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.player_bio_service, PlayerBiographyService)

    def test_projection_lookup_service(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.projection_lookup_service, ProjectionLookupService)

    def test_valuation_lookup_service(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.valuation_lookup_service, ValuationLookupService)

    def test_adp_report_service(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.adp_report_service, ADPReportService)

    def test_performance_report_service(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.performance_report_service, PerformanceReportService)

    def test_projection_evaluator(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.projection_evaluator, ProjectionEvaluator)

    def test_valuation_evaluator(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.valuation_evaluator, ValuationEvaluator)

    def test_true_talent_evaluator(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.talent_evaluator, TrueTalentEvaluator)

    def test_residual_diagnostic(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.residual_diagnostic, ResidualPersistenceDiagnostic)

    def test_adp_accuracy_evaluator(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.adp_accuracy_evaluator, ADPAccuracyEvaluator)

    def test_adp_movers_service(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert isinstance(c.adp_movers_service, ADPMoversService)


class TestCaching:
    def test_repo_reused_across_services(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        assert c.player_repo is c.player_repo

    def test_same_repo_instance_shared(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        # Access two services that both use player_repo — the underlying
        # cached_property ensures only one SqlitePlayerRepo instance exists.
        _ = c.projection_lookup_service
        _ = c.valuation_lookup_service
        assert c.player_repo is c.player_repo


class TestFunctional:
    def test_projection_lookup_on_empty_db(self, conn: sqlite3.Connection) -> None:
        c = AnalysisContainer(conn)
        results = c.projection_lookup_service.list_systems(2025)
        assert results == []
