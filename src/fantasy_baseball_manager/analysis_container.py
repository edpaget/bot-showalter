from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from fantasy_baseball_manager.repos import (
    SqliteADPRepo,
    SqliteBattingStatsRepo,
    SqlitePitchingStatsRepo,
    SqlitePlayerRepo,
    SqlitePositionAppearanceRepo,
    SqliteProjectionRepo,
    SqliteRosterStintRepo,
    SqliteTeamRepo,
    SqliteValuationRepo,
)
from fantasy_baseball_manager.services import (
    ADPAccuracyEvaluator,
    ADPMoversService,
    ADPReportService,
    PerformanceReportService,
    PlayerBiographyService,
    PlayerProfileService,
    ProjectionEvaluator,
    ProjectionLookupService,
    ResidualAnalysisDiagnostic,
    ResidualAnalyzer,
    ResidualPersistenceDiagnostic,
    StatsBasedPlayerUniverse,
    TrueTalentEvaluator,
    ValuationEvaluator,
    ValuationLookupService,
)

if TYPE_CHECKING:
    import sqlite3


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
    def player_profile_service(self) -> PlayerProfileService:
        return PlayerProfileService(player_repo=self.player_repo)

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
    def residual_analysis_diagnostic(self) -> ResidualAnalysisDiagnostic:
        return ResidualAnalysisDiagnostic(
            self.projection_repo,
            self.batting_stats_repo,
            self.pitching_stats_repo,
            self.player_repo,
        )

    @functools.cached_property
    def residual_analyzer(self) -> ResidualAnalyzer:
        return ResidualAnalyzer(
            self.projection_repo,
            self.batting_stats_repo,
            self.pitching_stats_repo,
            self.player_repo,
            self.position_appearance_repo,
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

    @functools.cached_property
    def player_universe(self) -> StatsBasedPlayerUniverse:
        return StatsBasedPlayerUniverse(
            batting_repo=self.batting_stats_repo,
            pitching_repo=self.pitching_stats_repo,
        )
