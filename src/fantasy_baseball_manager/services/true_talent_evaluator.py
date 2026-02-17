from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.talent_quality import (
    StatTalentMetrics,
    TalentQualitySummary,
    TrueTalentQualityReport,
    compute_predictive_validity,
    compute_r_squared_with_buckets,
    compute_residual_yoy_correlation,
    compute_shrinkage,
)
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.repos.protocols import BattingStatsRepo, PitchingStatsRepo, ProjectionRepo
from fantasy_baseball_manager.services.performance_report import _get_batter_actual, _get_pitcher_actual

_BATTER_BUCKET_EDGES = (200.0, 400.0)
_BATTER_BUCKET_LABELS = ("<200", "200-400", "400+")
_PITCHER_BUCKET_EDGES = (50.0, 120.0)
_PITCHER_BUCKET_LABELS = ("<50", "50-120", "120+")


class TrueTalentEvaluator:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        batting_repo: BattingStatsRepo,
        pitching_repo: PitchingStatsRepo,
    ) -> None:
        self._projection_repo = projection_repo
        self._batting_repo = batting_repo
        self._pitching_repo = pitching_repo

    def evaluate(
        self,
        system: str,
        version: str,
        season_n: int,
        season_n1: int,
        player_type: str,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
    ) -> TrueTalentQualityReport:
        projections = self._projection_repo.get_by_system_version(system, version)

        projs_n: dict[int, dict[str, float]] = {}
        projs_n1: dict[int, dict[str, float]] = {}
        for p in projections:
            if p.player_type != player_type:
                continue
            if p.season == season_n:
                projs_n[p.player_id] = {k: float(v) for k, v in p.stat_json.items() if isinstance(v, int | float)}
            elif p.season == season_n1:
                projs_n1[p.player_id] = {k: float(v) for k, v in p.stat_json.items() if isinstance(v, int | float)}

        is_pitcher = player_type == "pitcher"
        if is_pitcher:
            actuals_n_list = self._pitching_repo.get_by_season(season_n, source=actuals_source)
            actuals_n1_list = self._pitching_repo.get_by_season(season_n1, source=actuals_source)
            actuals_n_by_pid: dict[int, BattingStats | PitchingStats] = {a.player_id: a for a in actuals_n_list}
            actuals_n1_by_pid: dict[int, BattingStats | PitchingStats] = {a.player_id: a for a in actuals_n1_list}
            bucket_edges = _PITCHER_BUCKET_EDGES
            bucket_labels = _PITCHER_BUCKET_LABELS
        else:
            bat_n_list = self._batting_repo.get_by_season(season_n, source=actuals_source)
            bat_n1_list = self._batting_repo.get_by_season(season_n1, source=actuals_source)
            actuals_n_by_pid = {a.player_id: a for a in bat_n_list}
            actuals_n1_by_pid = {a.player_id: a for a in bat_n1_list}
            bucket_edges = _BATTER_BUCKET_EDGES
            bucket_labels = _BATTER_BUCKET_LABELS

        target_stats: tuple[str, ...] | list[str]
        if stats is not None:
            target_stats = stats
        elif is_pitcher:
            target_stats = PITCHER_TARGETS
        else:
            target_stats = BATTER_TARGETS

        get_actual = _get_pitcher_actual if is_pitcher else _get_batter_actual

        stat_metrics_list: list[StatTalentMetrics] = []

        for stat_name in target_stats:
            # Build aligned lists for season N
            estimates_n: list[float] = []
            raw_actuals_n: list[float] = []
            sample_sizes_n: list[float] = []
            pids_n: list[int] = []

            for pid, proj_stats in projs_n.items():
                est = proj_stats.get(stat_name)
                if est is None:
                    continue
                actual_obj = actuals_n_by_pid.get(pid)
                if actual_obj is None:
                    continue
                raw_val = get_actual(actual_obj, stat_name)  # type: ignore[arg-type]
                if raw_val is None:
                    continue
                ss = _get_sample_size(actual_obj, is_pitcher)
                estimates_n.append(est)
                raw_actuals_n.append(raw_val)
                sample_sizes_n.append(ss)
                pids_n.append(pid)

            n_season_n = len(pids_n)

            # Build aligned lists for returning players (present in both seasons)
            model_est_n_returning: list[float] = []
            raw_n_returning: list[float] = []
            actuals_n1_returning: list[float] = []
            residuals_n_returning: list[float] = []
            residuals_n1_returning: list[float] = []

            for i, pid in enumerate(pids_n):
                # Need N+1 actuals
                actual_n1_obj = actuals_n1_by_pid.get(pid)
                if actual_n1_obj is None:
                    continue
                actual_n1_val = get_actual(actual_n1_obj, stat_name)  # type: ignore[arg-type]
                if actual_n1_val is None:
                    continue

                # Need N+1 projection for residual computation
                proj_n1_stats = projs_n1.get(pid)
                if proj_n1_stats is None:
                    continue
                est_n1 = proj_n1_stats.get(stat_name)
                if est_n1 is None:
                    continue

                model_est_n_returning.append(estimates_n[i])
                raw_n_returning.append(raw_actuals_n[i])
                actuals_n1_returning.append(actual_n1_val)
                residuals_n_returning.append(raw_actuals_n[i] - estimates_n[i])
                residuals_n1_returning.append(actual_n1_val - est_n1)

            n_returning = len(model_est_n_returning)

            # Compute metrics
            model_corr, raw_corr = compute_predictive_validity(
                model_est_n_returning, raw_n_returning, actuals_n1_returning
            )
            residual_yoy = compute_residual_yoy_correlation(residuals_n_returning, residuals_n1_returning)
            shrinkage_ratio, est_raw_corr = compute_shrinkage(estimates_n, raw_actuals_n)
            r_squared, residual_by_bucket = compute_r_squared_with_buckets(
                estimates_n, raw_actuals_n, sample_sizes_n, bucket_edges, bucket_labels
            )
            regression_rate = 1.0 - residual_yoy

            stat_metrics_list.append(
                StatTalentMetrics(
                    stat_name=stat_name,
                    model_next_season_corr=model_corr,
                    raw_next_season_corr=raw_corr,
                    predictive_validity_pass=model_corr > raw_corr,
                    residual_yoy_corr=residual_yoy,
                    residual_non_persistence_pass=abs(residual_yoy) < 0.15,
                    shrinkage_ratio=shrinkage_ratio,
                    estimate_raw_corr=est_raw_corr,
                    shrinkage_pass=shrinkage_ratio < 0.9,
                    r_squared=r_squared,
                    residual_by_bucket=residual_by_bucket,
                    r_squared_pass=r_squared > 0.7,
                    regression_rate=regression_rate,
                    regression_rate_pass=regression_rate > 0.80,
                    n_season_n=n_season_n,
                    n_returning=n_returning,
                )
            )

        summary = TalentQualitySummary(
            predictive_validity_passes=sum(1 for m in stat_metrics_list if m.predictive_validity_pass),
            predictive_validity_total=len(stat_metrics_list),
            residual_non_persistence_passes=sum(1 for m in stat_metrics_list if m.residual_non_persistence_pass),
            residual_non_persistence_total=len(stat_metrics_list),
            shrinkage_passes=sum(1 for m in stat_metrics_list if m.shrinkage_pass),
            shrinkage_total=len(stat_metrics_list),
            r_squared_passes=sum(1 for m in stat_metrics_list if m.r_squared_pass),
            r_squared_total=len(stat_metrics_list),
            regression_rate_passes=sum(1 for m in stat_metrics_list if m.regression_rate_pass),
            regression_rate_total=len(stat_metrics_list),
        )

        return TrueTalentQualityReport(
            system=system,
            version=version,
            season_n=season_n,
            season_n1=season_n1,
            player_type=player_type,
            stat_metrics=stat_metrics_list,
            summary=summary,
        )


def _get_sample_size(actual: BattingStats | PitchingStats, is_pitcher: bool) -> float:
    if is_pitcher and isinstance(actual, PitchingStats):
        return float(actual.ip) if actual.ip is not None else 0.0
    if isinstance(actual, BattingStats):
        return float(actual.pa) if actual.pa is not None else 0.0
    return 0.0
