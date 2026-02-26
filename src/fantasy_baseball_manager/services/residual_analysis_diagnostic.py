import statistics
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain.residual_analysis import (
    ResidualAnalysisReport,
    ResidualAnalysisSummary,
    StatResidualAnalysis,
    compute_calibration_bins,
    compute_heteroscedasticity,
    compute_mean_bias,
)
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.services.performance_report import _get_batter_actual, _get_pitcher_actual

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.batting_stats import BattingStats
    from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
    from fantasy_baseball_manager.repos.protocols import (
        BattingStatsRepo,
        PitchingStatsRepo,
        PlayerRepo,
        ProjectionRepo,
    )


def _top_by_war_batting(items: list[BattingStats], top: int) -> list[BattingStats]:
    return sorted(items, key=lambda x: x.war if x.war is not None else 0.0, reverse=True)[:top]


def _top_by_war_pitching(items: list[PitchingStats], top: int) -> list[PitchingStats]:
    return sorted(items, key=lambda x: x.war if x.war is not None else 0.0, reverse=True)[:top]


class ResidualAnalysisDiagnostic:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
        player_repo: PlayerRepo,
    ) -> None:
        self._projection_repo = projection_repo
        self._batting_repo = batting_repo
        self._pitching_repo = pitching_repo
        self._player_repo = player_repo

    def analyze(
        self,
        system: str,
        version: str,
        seasons: list[int],
        stats: list[str] | None = None,
        top: int | None = None,
        actuals_source: str = "fangraphs",
        min_pa: int | None = None,
        min_ip: int | None = None,
    ) -> ResidualAnalysisReport:
        projections = self._projection_repo.get_by_system_version(system, version)

        # Group projections by (season, player_type)
        batter_projs: dict[int, dict[int, dict[str, float]]] = {}  # season -> pid -> stats
        pitcher_projs: dict[int, dict[int, dict[str, float]]] = {}
        for p in projections:
            if p.season not in seasons:
                continue
            stat_dict = {k: float(v) for k, v in p.stat_json.items() if isinstance(v, int | float)}
            if p.player_type == "batter":
                batter_projs.setdefault(p.season, {})[p.player_id] = stat_dict
            elif p.player_type == "pitcher":
                pitcher_projs.setdefault(p.season, {})[p.player_id] = stat_dict

        # Load actuals, filtered by top-N actual WAR and min PA/IP
        batter_actuals: dict[int, list[BattingStats]] = {}
        pitcher_actuals: dict[int, list[PitchingStats]] = {}
        for season in seasons:
            bat_list = self._batting_repo.get_by_season(season, source=actuals_source)
            if top is not None:
                bat_list = _top_by_war_batting(bat_list, top)
            if min_pa is not None:
                bat_list = [a for a in bat_list if (a.pa or 0) >= min_pa]
            batter_actuals[season] = bat_list

            pit_list = self._pitching_repo.get_by_season(season, source=actuals_source)
            if top is not None:
                pit_list = _top_by_war_pitching(pit_list, top)
            if min_ip is not None:
                pit_list = [a for a in pit_list if (a.ip or 0) >= min_ip]
            pitcher_actuals[season] = pit_list

        # Determine target stats
        batter_stats = list(stats) if stats else list(BATTER_TARGETS)
        pitcher_stats = list(stats) if stats else list(PITCHER_TARGETS)

        stat_analyses: list[StatResidualAnalysis] = []

        # Analyze batters
        for stat_name in batter_stats:
            analysis = self._analyze_stat(
                stat_name=stat_name,
                player_type="batter",
                projs_by_season=batter_projs,
                actuals_by_season=batter_actuals,
                get_actual=_get_batter_actual,
            )
            if analysis is not None:
                stat_analyses.append(analysis)

        # Analyze pitchers
        for stat_name in pitcher_stats:
            analysis = self._analyze_stat(
                stat_name=stat_name,
                player_type="pitcher",
                projs_by_season=pitcher_projs,
                actuals_by_season=pitcher_actuals,
                get_actual=_get_pitcher_actual,
            )
            if analysis is not None:
                stat_analyses.append(analysis)

        # Build summary
        n_bias_significant = sum(1 for a in stat_analyses if a.bias_significant)
        n_hetero_significant = sum(1 for a in stat_analyses if a.heteroscedasticity_significant)
        n_total = len(stat_analyses)

        summary = ResidualAnalysisSummary(
            n_bias_significant=n_bias_significant,
            n_bias_total=n_total,
            n_hetero_significant=n_hetero_significant,
            n_hetero_total=n_total,
            calibration_recommended=n_bias_significant > 0,
        )

        return ResidualAnalysisReport(
            system=system,
            version=version,
            seasons=seasons,
            top=top,
            stat_analyses=stat_analyses,
            summary=summary,
        )

    def _analyze_stat(
        self,
        stat_name: str,
        player_type: str,
        projs_by_season: dict[int, dict[int, dict[str, float]]],
        actuals_by_season: dict[int, list[BattingStats]] | dict[int, list[PitchingStats]],
        get_actual: Any,
    ) -> StatResidualAnalysis | None:
        predictions: list[float] = []
        actuals: list[float] = []

        for season, projs in projs_by_season.items():
            season_actuals = actuals_by_season.get(season, [])

            for actual_obj in season_actuals:
                pid = actual_obj.player_id
                proj_stats = projs.get(pid)
                if proj_stats is None:
                    continue
                est = proj_stats.get(stat_name)
                if est is None:
                    continue
                actual_val = get_actual(actual_obj, stat_name)
                if actual_val is None:
                    continue
                predictions.append(est)
                actuals.append(actual_val)

        if not predictions:
            return None

        residuals = [actuals[i] - predictions[i] for i in range(len(predictions))]
        abs_residuals = [abs(r) for r in residuals]

        mean_residual, bias_significant = compute_mean_bias(residuals)
        hetero_corr, hetero_significant = compute_heteroscedasticity(predictions, abs_residuals)
        calibration_bins = compute_calibration_bins(predictions, actuals)

        try:
            std_residual = statistics.stdev(residuals) if len(residuals) >= 2 else 0.0
        except statistics.StatisticsError:
            std_residual = 0.0

        return StatResidualAnalysis(
            stat_name=stat_name,
            player_type=player_type,
            n_observations=len(predictions),
            mean_residual=mean_residual,
            std_residual=std_residual,
            bias_significant=bias_significant,
            heteroscedasticity_corr=hetero_corr,
            heteroscedasticity_significant=hetero_significant,
            calibration_bins=calibration_bins,
        )
