from __future__ import annotations

import functools
import sqlite3

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


class AnalysisContainer:
    """DI container for analysis services shared by the CLI and future agent."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # --- Repos ---

    @functools.cached_property
    def player_repo(self) -> SqlitePlayerRepo:
        return SqlitePlayerRepo(self._conn)

    @functools.cached_property
    def team_repo(self) -> SqliteTeamRepo:
        return SqliteTeamRepo(self._conn)

    @functools.cached_property
    def projection_repo(self) -> SqliteProjectionRepo:
        return SqliteProjectionRepo(self._conn)

    @functools.cached_property
    def valuation_repo(self) -> SqliteValuationRepo:
        return SqliteValuationRepo(self._conn)

    @functools.cached_property
    def adp_repo(self) -> SqliteADPRepo:
        return SqliteADPRepo(self._conn)

    @functools.cached_property
    def batting_stats_repo(self) -> SqliteBattingStatsRepo:
        return SqliteBattingStatsRepo(self._conn)

    @functools.cached_property
    def pitching_stats_repo(self) -> SqlitePitchingStatsRepo:
        return SqlitePitchingStatsRepo(self._conn)

    @functools.cached_property
    def position_appearance_repo(self) -> SqlitePositionAppearanceRepo:
        return SqlitePositionAppearanceRepo(self._conn)

    @functools.cached_property
    def roster_stint_repo(self) -> SqliteRosterStintRepo:
        return SqliteRosterStintRepo(self._conn)

    # --- Services ---

    @functools.cached_property
    def player_bio_service(self) -> PlayerBiographyService:
        return PlayerBiographyService(
            player_repo=self.player_repo,
            team_repo=self.team_repo,
            roster_stint_repo=self.roster_stint_repo,
            batting_stats_repo=self.batting_stats_repo,
            pitching_stats_repo=self.pitching_stats_repo,
            position_appearance_repo=self.position_appearance_repo,
        )

    @functools.cached_property
    def projection_lookup_service(self) -> ProjectionLookupService:
        return ProjectionLookupService(self.player_repo, self.projection_repo)

    @functools.cached_property
    def valuation_lookup_service(self) -> ValuationLookupService:
        return ValuationLookupService(self.player_repo, self.valuation_repo)

    @functools.cached_property
    def adp_report_service(self) -> ADPReportService:
        return ADPReportService(self.player_repo, self.valuation_repo, self.adp_repo)

    @functools.cached_property
    def adp_movers_service(self) -> ADPMoversService:
        return ADPMoversService(self.adp_repo, self.player_repo)

    @functools.cached_property
    def performance_report_service(self) -> PerformanceReportService:
        return PerformanceReportService(
            self.projection_repo,
            self.player_repo,
            self.batting_stats_repo,
            self.pitching_stats_repo,
        )

    @functools.cached_property
    def projection_evaluator(self) -> ProjectionEvaluator:
        return ProjectionEvaluator(
            self.projection_repo,
            self.batting_stats_repo,
            self.pitching_stats_repo,
        )

    @functools.cached_property
    def valuation_evaluator(self) -> ValuationEvaluator:
        return ValuationEvaluator(
            valuation_repo=self.valuation_repo,
            batting_repo=self.batting_stats_repo,
            pitching_repo=self.pitching_stats_repo,
            position_repo=self.position_appearance_repo,
            player_repo=self.player_repo,
        )

    @functools.cached_property
    def talent_evaluator(self) -> TrueTalentEvaluator:
        return TrueTalentEvaluator(
            self.projection_repo,
            self.batting_stats_repo,
            self.pitching_stats_repo,
        )

    @functools.cached_property
    def residual_diagnostic(self) -> ResidualPersistenceDiagnostic:
        return ResidualPersistenceDiagnostic(
            self.projection_repo,
            self.batting_stats_repo,
            self.player_repo,
        )

    @functools.cached_property
    def adp_accuracy_evaluator(self) -> ADPAccuracyEvaluator:
        return ADPAccuracyEvaluator(
            adp_repo=self.adp_repo,
            valuation_repo=self.valuation_repo,
            player_repo=self.player_repo,
            batting_repo=self.batting_stats_repo,
            pitching_repo=self.pitching_stats_repo,
            position_repo=self.position_appearance_repo,
        )
