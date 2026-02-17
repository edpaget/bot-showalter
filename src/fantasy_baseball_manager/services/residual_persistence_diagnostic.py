from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.residual_persistence import (
    ResidualPersistenceReport,
    ResidualPersistenceSummary,
    StatResidualPersistence,
    compute_residual_correlation_by_bucket,
    compute_rmse_ceiling,
    identify_chronic_performers,
)
from fantasy_baseball_manager.domain.talent_quality import compute_residual_yoy_correlation
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS
from fantasy_baseball_manager.repos.protocols import BattingStatsRepo, PlayerRepo, ProjectionRepo
from fantasy_baseball_manager.services.performance_report import _get_batter_actual

_BUCKET_EDGES = (200.0, 400.0)
_BUCKET_LABELS = ("<200", "200-400", "400+")


class ResidualPersistenceDiagnostic:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        batting_repo: BattingStatsRepo,
        player_repo: PlayerRepo,
    ) -> None:
        self._projection_repo = projection_repo
        self._batting_repo = batting_repo
        self._player_repo = player_repo

    def diagnose(
        self,
        system: str,
        version: str,
        season_n: int,
        season_n1: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
    ) -> ResidualPersistenceReport:
        projections = self._projection_repo.get_by_system_version(system, version)

        projs_n: dict[int, dict[str, float]] = {}
        projs_n1: dict[int, dict[str, float]] = {}
        for p in projections:
            if p.player_type != "batter":
                continue
            if p.season == season_n:
                projs_n[p.player_id] = {k: float(v) for k, v in p.stat_json.items() if isinstance(v, int | float)}
            elif p.season == season_n1:
                projs_n1[p.player_id] = {k: float(v) for k, v in p.stat_json.items() if isinstance(v, int | float)}

        bat_n_list = self._batting_repo.get_by_season(season_n, source=actuals_source)
        bat_n1_list = self._batting_repo.get_by_season(season_n1, source=actuals_source)
        actuals_n_by_pid: dict[int, BattingStats] = {a.player_id: a for a in bat_n_list}
        actuals_n1_by_pid: dict[int, BattingStats] = {a.player_id: a for a in bat_n1_list}

        players = self._player_repo.all()
        player_by_id: dict[int, Player] = {p.id: p for p in players if p.id is not None}

        target_stats: tuple[str, ...] | list[str]
        if stats is not None:
            target_stats = stats
        else:
            target_stats = BATTER_TARGETS

        stat_metrics_list: list[StatResidualPersistence] = []

        for stat_name in target_stats:
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
                raw_val = _get_batter_actual(actual_obj, stat_name)
                if raw_val is None:
                    continue
                ss = float(actual_obj.pa) if actual_obj.pa is not None else 0.0
                estimates_n.append(est)
                raw_actuals_n.append(raw_val)
                sample_sizes_n.append(ss)
                pids_n.append(pid)

            # Build aligned lists for returning players
            residuals_n_ret: list[float] = []
            residuals_n1_ret: list[float] = []
            pids_ret: list[int] = []
            names_ret: list[str] = []
            pa_n_ret: list[float] = []
            pa_n1_ret: list[float] = []
            actuals_n1_ret: list[float] = []
            estimates_n1_ret: list[float] = []
            sample_sizes_ret: list[float] = []

            for i, pid in enumerate(pids_n):
                actual_n1_obj = actuals_n1_by_pid.get(pid)
                if actual_n1_obj is None:
                    continue
                actual_n1_val = _get_batter_actual(actual_n1_obj, stat_name)
                if actual_n1_val is None:
                    continue

                proj_n1_stats = projs_n1.get(pid)
                if proj_n1_stats is None:
                    continue
                est_n1 = proj_n1_stats.get(stat_name)
                if est_n1 is None:
                    continue

                res_n = raw_actuals_n[i] - estimates_n[i]
                res_n1 = actual_n1_val - est_n1

                player = player_by_id.get(pid)
                name = f"{player.name_first} {player.name_last}" if player else str(pid)

                pa_n1_val = float(actual_n1_obj.pa) if actual_n1_obj.pa is not None else 0.0

                residuals_n_ret.append(res_n)
                residuals_n1_ret.append(res_n1)
                pids_ret.append(pid)
                names_ret.append(name)
                pa_n_ret.append(sample_sizes_n[i])
                pa_n1_ret.append(pa_n1_val)
                actuals_n1_ret.append(actual_n1_val)
                estimates_n1_ret.append(est_n1)
                sample_sizes_ret.append(sample_sizes_n[i])

            n_returning = len(pids_ret)

            # Compute metrics
            overall_corr = compute_residual_yoy_correlation(residuals_n_ret, residuals_n1_ret)

            corr_by_bucket, n_by_bucket = compute_residual_correlation_by_bucket(
                residuals_n_ret, residuals_n1_ret, sample_sizes_ret, _BUCKET_EDGES, _BUCKET_LABELS
            )

            overperformers, underperformers = identify_chronic_performers(
                residuals_n_ret, residuals_n1_ret, pids_ret, names_ret, pa_n_ret, pa_n1_ret
            )

            rmse_baseline, rmse_corrected, improvement_pct = compute_rmse_ceiling(
                actuals_n1_ret, estimates_n1_ret, residuals_n_ret
            )

            stat_metrics_list.append(
                StatResidualPersistence(
                    stat_name=stat_name,
                    residual_corr_overall=overall_corr,
                    residual_corr_by_bucket=corr_by_bucket,
                    n_by_bucket=n_by_bucket,
                    chronic_overperformers=overperformers,
                    chronic_underperformers=underperformers,
                    rmse_baseline=rmse_baseline,
                    rmse_corrected=rmse_corrected,
                    rmse_improvement_pct=improvement_pct,
                    persistence_pass=overall_corr > 0.10,
                    ceiling_pass=improvement_pct > 2.0,
                    n_returning=n_returning,
                )
            )

        persistence_passes = sum(1 for m in stat_metrics_list if m.persistence_pass)
        ceiling_passes = sum(1 for m in stat_metrics_list if m.ceiling_pass)
        total = len(stat_metrics_list)

        summary = ResidualPersistenceSummary(
            persistence_passes=persistence_passes,
            persistence_total=total,
            ceiling_passes=ceiling_passes,
            ceiling_total=total,
            go=persistence_passes >= 3 and ceiling_passes >= 2,
        )

        return ResidualPersistenceReport(
            system=system,
            version=version,
            season_n=season_n,
            season_n1=season_n1,
            stat_metrics=stat_metrics_list,
            summary=summary,
        )
