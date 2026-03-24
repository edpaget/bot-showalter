from __future__ import annotations

import math
import statistics
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

from scipy.stats import ConstantInputWarning, pearsonr, spearmanr

from fantasy_baseball_manager.domain import (
    BinnedValue,
    BinTargetMean,
    ColumnProfile,
    CorrelationScanResult,
    MultiColumnRanking,
    PlayerType,
    PooledCorrelationResult,
    SeasonCorrelationResult,
    StabilityResult,
    TargetCorrelation,
    TargetStability,
)
from fantasy_baseball_manager.models.composite.targets import BATTER_TARGETS, PITCHER_TARGETS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.repos import ConnectionProvider

NUMERIC_COLUMNS: tuple[str, ...] = (
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "barrel",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
    "hc_x",
    "hc_y",
    "release_extension",
)

_VALID_PLAYER_TYPES = {"batter", "pitcher"}


class StatcastColumnProfiler:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def profile_columns(
        self,
        columns: Sequence[str],
        seasons: Sequence[int],
        player_type: str,
    ) -> list[ColumnProfile]:
        """Profile the given columns, returning one ColumnProfile per (column, season)."""
        if player_type not in _VALID_PLAYER_TYPES:
            msg = f"player_type must be 'batter' or 'pitcher', got '{player_type}'"
            raise ValueError(msg)

        invalid = [c for c in columns if c not in NUMERIC_COLUMNS]
        if invalid:
            msg = f"Invalid column(s): {', '.join(invalid)}. Must be one of {NUMERIC_COLUMNS}"
            raise ValueError(msg)

        player_col = "batter_id" if player_type == "batter" else "pitcher_id"
        results: list[ColumnProfile] = []

        for column in columns:
            rows = self._fetch_aggregated(column, player_col, seasons)
            by_season: dict[int, list[tuple[float | None, int]]] = defaultdict(list)
            for _player_id, season, avg_val, pitch_count in rows:
                by_season[season].append((avg_val, pitch_count))

            for season in sorted(seasons):
                season_data = by_season.get(season, [])
                results.append(self._compute_profile(column, season, player_type, season_data))

        return results

    def _fetch_aggregated(
        self,
        column: str,
        player_col: str,
        seasons: Sequence[int],
    ) -> list[tuple[int, int, float | None, int]]:
        """Fetch per-player-season aggregated values."""
        placeholders = ",".join("?" for _ in seasons)
        sql = f"""
            SELECT {player_col} AS player_id,
                   CAST(SUBSTR(game_date, 1, 4) AS INTEGER) AS season,
                   AVG({column}) AS avg_val,
                   COUNT(*) AS pitch_count
            FROM statcast_pitch
            WHERE CAST(SUBSTR(game_date, 1, 4) AS INTEGER) IN ({placeholders})
            GROUP BY {player_col}, season
        """  # noqa: S608
        with self._provider.connection() as conn:
            cursor = conn.execute(sql, list(seasons))
            return cursor.fetchall()

    @staticmethod
    def _compute_profile(
        column: str,
        season: int,
        player_type: str,
        data: list[tuple[float | None, int]],
    ) -> ColumnProfile:
        """Compute distribution statistics from aggregated player-season values."""
        non_null = [val for val, _ in data if val is not None]
        null_count = len(data) - len(non_null)
        total = len(data)
        null_pct = (null_count / total * 100) if total > 0 else 0.0

        if not non_null:
            return ColumnProfile(
                column=column,
                season=season,
                player_type=PlayerType(player_type),
                count=0,
                null_count=null_count,
                null_pct=null_pct,
                mean=0.0,
                median=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                p10=0.0,
                p25=0.0,
                p75=0.0,
                p90=0.0,
                skewness=0.0,
            )

        count = len(non_null)
        mean = statistics.mean(non_null)
        median = statistics.median(non_null)
        std = statistics.stdev(non_null) if count > 1 else 0.0
        min_val = min(non_null)
        max_val = max(non_null)

        if count >= 2:
            quartiles = statistics.quantiles(non_null, n=4)
            deciles = statistics.quantiles(non_null, n=10)
            p10 = deciles[0]
            p25 = quartiles[0]
            p75 = quartiles[2]
            p90 = deciles[8]
        else:
            p10 = p25 = p75 = p90 = non_null[0]

        skewness = _compute_skewness(non_null, mean, std)

        return ColumnProfile(
            column=column,
            season=season,
            player_type=PlayerType(player_type),
            count=count,
            null_count=null_count,
            null_pct=null_pct,
            mean=mean,
            median=median,
            std=std,
            min=min_val,
            max=max_val,
            p10=p10,
            p25=p25,
            p75=p75,
            p90=p90,
            skewness=skewness,
        )


def _compute_skewness(values: list[float], mean: float, std: float) -> float:
    """Compute Fisher's skewness: mean((x - mean)^3) / std^3."""
    if std == 0.0 or len(values) < 3:
        return 0.0
    n = len(values)
    m3 = sum((x - mean) ** 3 for x in values) / n
    return m3 / (std**3)


def _compute_correlation_pair(candidate: list[float], target: list[float]) -> tuple[float, float, float, float]:
    """Compute Pearson and Spearman correlations, handling edge cases."""
    n = len(candidate)
    if n < 3:
        return 0.0, 1.0, 0.0, 1.0

    # Pearson — catch constant-input and degenerate-data errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        try:
            p_corr, p_pval = pearsonr(candidate, target)
            p_r: float = float(p_corr)
            p_p: float = float(p_pval)
        except Exception:  # noqa: BLE001
            p_r, p_p = 0.0, 1.0
    if math.isnan(p_r):
        p_r, p_p = 0.0, 1.0

    # Spearman
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        s_corr, s_pval = spearmanr(candidate, target)
    s_r: float = float(s_corr)
    s_p: float = float(s_pval)
    if math.isnan(s_r):
        s_r, s_p = 0.0, 1.0

    return p_r, p_p, s_r, s_p


class CorrelationScanner:
    """Compute correlations between statcast columns and model targets."""

    def __init__(self, statcast_provider: ConnectionProvider, stats_provider: ConnectionProvider) -> None:
        self._sc = statcast_provider
        self._st = stats_provider

    def scan_target_correlations(
        self,
        column_spec: str,
        seasons: Sequence[int],
        player_type: str,
    ) -> CorrelationScanResult:
        """Scan a single column spec against all targets for the given player type."""
        if player_type not in _VALID_PLAYER_TYPES:
            msg = f"player_type must be 'batter' or 'pitcher', got '{player_type}'"
            raise ValueError(msg)

        player_col = "batter_id" if player_type == "batter" else "pitcher_id"
        targets = BATTER_TARGETS if player_type == "batter" else PITCHER_TARGETS

        candidate_values = self._fetch_candidate_values(column_spec, player_col, seasons)
        mlbam_to_pid = self._fetch_mlbam_to_player_id()

        if player_type == "batter":
            target_values = self._fetch_batter_targets(seasons)
        else:
            target_values = self._fetch_pitcher_targets(seasons)

        # Per-season results
        per_season: list[SeasonCorrelationResult] = []
        for season in sorted(seasons):
            correlations = self._correlate_season(candidate_values, target_values, mlbam_to_pid, targets, season)
            per_season.append(
                SeasonCorrelationResult(
                    column_spec=column_spec,
                    season=season,
                    player_type=PlayerType(player_type),
                    correlations=tuple(correlations),
                )
            )

        # Pooled across all seasons
        pooled_correlations = self._correlate_pooled(candidate_values, target_values, mlbam_to_pid, targets, seasons)
        pooled = PooledCorrelationResult(
            column_spec=column_spec,
            player_type=PlayerType(player_type),
            correlations=tuple(pooled_correlations),
        )

        return CorrelationScanResult(
            column_spec=column_spec,
            player_type=PlayerType(player_type),
            per_season=tuple(per_season),
            pooled=pooled,
        )

    def scan_multiple(
        self,
        column_specs: Sequence[str],
        seasons: Sequence[int],
        player_type: str,
    ) -> list[CorrelationScanResult]:
        """Scan multiple column specs and return results for each."""
        return [self.scan_target_correlations(spec, seasons, player_type) for spec in column_specs]

    def scan_from_values(
        self,
        label: str,
        candidate_values: dict[tuple[int, int], float],
        seasons: Sequence[int],
        player_type: str,
    ) -> CorrelationScanResult:
        """Scan pre-computed candidate values against targets.

        Like ``scan_target_correlations`` but skips SQL aggregation —
        the caller provides the candidate dict directly.
        """
        if player_type not in _VALID_PLAYER_TYPES:
            msg = f"player_type must be 'batter' or 'pitcher', got '{player_type}'"
            raise ValueError(msg)

        targets = BATTER_TARGETS if player_type == "batter" else PITCHER_TARGETS
        mlbam_to_pid = self._fetch_mlbam_to_player_id()

        if player_type == "batter":
            target_values = self._fetch_batter_targets(seasons)
        else:
            target_values = self._fetch_pitcher_targets(seasons)

        per_season: list[SeasonCorrelationResult] = []
        for season in sorted(seasons):
            correlations = self._correlate_season(candidate_values, target_values, mlbam_to_pid, targets, season)
            per_season.append(
                SeasonCorrelationResult(
                    column_spec=label,
                    season=season,
                    player_type=PlayerType(player_type),
                    correlations=tuple(correlations),
                )
            )

        pooled_correlations = self._correlate_pooled(candidate_values, target_values, mlbam_to_pid, targets, seasons)
        pooled = PooledCorrelationResult(
            column_spec=label,
            player_type=PlayerType(player_type),
            correlations=tuple(pooled_correlations),
        )

        return CorrelationScanResult(
            column_spec=label,
            player_type=PlayerType(player_type),
            per_season=tuple(per_season),
            pooled=pooled,
        )

    def compute_bin_target_means(
        self,
        binned_values: list[BinnedValue],
        seasons: Sequence[int],
        player_type: str,
    ) -> list[BinTargetMean]:
        """Compute mean target values within each bin label.

        Groups binned values by bin_label, maps mlbam IDs to player IDs,
        looks up target values, and computes per-target means.
        Returns results sorted by (bin_label, target).
        """
        if player_type not in _VALID_PLAYER_TYPES:
            msg = f"player_type must be 'batter' or 'pitcher', got '{player_type}'"
            raise ValueError(msg)

        targets = BATTER_TARGETS if player_type == "batter" else PITCHER_TARGETS
        mlbam_to_pid = self._fetch_mlbam_to_player_id()

        if player_type == "batter":
            target_values = self._fetch_batter_targets(seasons)
        else:
            target_values = self._fetch_pitcher_targets(seasons)

        # Group binned values by bin_label
        by_bin: dict[str, list[BinnedValue]] = defaultdict(list)
        for bv in binned_values:
            by_bin[bv.bin_label].append(bv)

        results: list[BinTargetMean] = []
        for bin_label in sorted(by_bin):
            bin_members = by_bin[bin_label]
            # Collect target values for members of this bin
            target_accum: dict[str, list[float]] = {t: [] for t in targets}
            for bv in bin_members:
                pid = mlbam_to_pid.get(bv.player_id)
                if pid is None:
                    continue
                tvals = target_values.get((pid, bv.season))
                if tvals is None:
                    continue
                for t in targets:
                    if t in tvals:
                        target_accum[t].append(tvals[t])

            for t in sorted(targets):
                vals = target_accum[t]
                if vals:
                    results.append(
                        BinTargetMean(
                            bin_label=bin_label,
                            target=t,
                            mean=statistics.mean(vals),
                            count=len(vals),
                        )
                    )

        return results

    def _fetch_candidate_values(
        self,
        column_spec: str,
        player_col: str,
        seasons: Sequence[int],
    ) -> dict[tuple[int, int], float]:
        """Fetch player-season aggregated values from statcast."""
        placeholders = ",".join("?" for _ in seasons)

        if " WHERE " in column_spec:
            # Expression: "AVG(launch_speed) WHERE barrel = 1"
            select_expr, where_clause = column_spec.split(" WHERE ", 1)
            sql = f"""
                SELECT {player_col} AS mlbam_id,
                       CAST(SUBSTR(game_date, 1, 4) AS INTEGER) AS season,
                       {select_expr} AS val
                FROM statcast_pitch
                WHERE CAST(SUBSTR(game_date, 1, 4) AS INTEGER) IN ({placeholders})
                  AND {where_clause}
                GROUP BY {player_col}, season
                HAVING val IS NOT NULL
            """  # noqa: S608
        elif column_spec in NUMERIC_COLUMNS:
            sql = f"""
                SELECT {player_col} AS mlbam_id,
                       CAST(SUBSTR(game_date, 1, 4) AS INTEGER) AS season,
                       AVG({column_spec}) AS val
                FROM statcast_pitch
                WHERE CAST(SUBSTR(game_date, 1, 4) AS INTEGER) IN ({placeholders})
                GROUP BY {player_col}, season
                HAVING val IS NOT NULL
            """  # noqa: S608
        else:
            # Treat as raw expression (e.g., "AVG(launch_speed)")
            sql = f"""
                SELECT {player_col} AS mlbam_id,
                       CAST(SUBSTR(game_date, 1, 4) AS INTEGER) AS season,
                       {column_spec} AS val
                FROM statcast_pitch
                WHERE CAST(SUBSTR(game_date, 1, 4) AS INTEGER) IN ({placeholders})
                GROUP BY {player_col}, season
                HAVING val IS NOT NULL
            """  # noqa: S608

        with self._sc.connection() as sc:
            rows = sc.execute(sql, list(seasons)).fetchall()
        return {(row[0], row[1]): float(row[2]) for row in rows}

    def _fetch_mlbam_to_player_id(self) -> dict[int, int]:
        """Map mlbam_id -> player.id from stats db."""
        with self._st.connection() as st:
            rows = st.execute("SELECT id, mlbam_id FROM player WHERE mlbam_id IS NOT NULL").fetchall()
        return {row[1]: row[0] for row in rows}

    def _fetch_batter_targets(self, seasons: Sequence[int]) -> dict[tuple[int, int], dict[str, float]]:
        """Fetch batter target values, computing derived targets."""
        placeholders = ",".join("?" for _ in seasons)
        sql = f"""
            SELECT player_id, season, avg, obp, slg, woba, h, hr, ab, so, sf
            FROM batting_stats
            WHERE source = 'fangraphs'
              AND season IN ({placeholders})
        """  # noqa: S608
        with self._st.connection() as st:
            rows = st.execute(sql, list(seasons)).fetchall()
        result: dict[tuple[int, int], dict[str, float]] = {}
        for row in rows:
            pid, season = row[0], row[1]
            avg_val, obp_val, slg_val, woba_val = row[2], row[3], row[4], row[5]
            h_val, hr_val, ab_val, so_val, sf_val = row[6], row[7], row[8], row[9], row[10]

            targets: dict[str, float] = {}
            if avg_val is not None:
                targets["avg"] = float(avg_val)
            if obp_val is not None:
                targets["obp"] = float(obp_val)
            if slg_val is not None:
                targets["slg"] = float(slg_val)
            if woba_val is not None:
                targets["woba"] = float(woba_val)

            # Derived: iso = slg - avg
            if slg_val is not None and avg_val is not None:
                targets["iso"] = float(slg_val) - float(avg_val)

            # Derived: babip = (h - hr) / (ab - so - hr + sf)
            if all(v is not None for v in (h_val, hr_val, ab_val, so_val, sf_val)):
                denom = ab_val - so_val - hr_val + sf_val
                if denom != 0:
                    targets["babip"] = (h_val - hr_val) / denom

            result[(pid, season)] = targets
        return result

    def _fetch_pitcher_targets(self, seasons: Sequence[int]) -> dict[tuple[int, int], dict[str, float]]:
        """Fetch pitcher target values, computing derived targets."""
        placeholders = ",".join("?" for _ in seasons)
        sql = f"""
            SELECT player_id, season, era, fip, k_per_9, bb_per_9, whip, hr, ip, h, so
            FROM pitching_stats
            WHERE source = 'fangraphs'
              AND season IN ({placeholders})
        """  # noqa: S608
        with self._st.connection() as st:
            rows = st.execute(sql, list(seasons)).fetchall()
        result: dict[tuple[int, int], dict[str, float]] = {}
        for row in rows:
            pid, season = row[0], row[1]
            era_val, fip_val, k9_val, bb9_val, whip_val = row[2], row[3], row[4], row[5], row[6]
            hr_val, ip_val, h_val, so_val = row[7], row[8], row[9], row[10]

            targets: dict[str, float] = {}
            if era_val is not None:
                targets["era"] = float(era_val)
            if fip_val is not None:
                targets["fip"] = float(fip_val)
            if k9_val is not None:
                targets["k_per_9"] = float(k9_val)
            if bb9_val is not None:
                targets["bb_per_9"] = float(bb9_val)
            if whip_val is not None:
                targets["whip"] = float(whip_val)

            # Derived: hr_per_9 = hr * 9 / ip
            if hr_val is not None and ip_val is not None and ip_val != 0:
                targets["hr_per_9"] = float(hr_val) * 9 / float(ip_val)

            # Derived: babip = (h - hr) / (ip * 3 + h - so - hr)
            if all(v is not None for v in (h_val, hr_val, ip_val, so_val)):
                denom = float(ip_val) * 3 + h_val - so_val - hr_val
                if denom != 0:
                    targets["babip"] = (h_val - hr_val) / denom

            result[(pid, season)] = targets
        return result

    def _correlate_season(
        self,
        candidate_values: dict[tuple[int, int], float],
        target_values: dict[tuple[int, int], dict[str, float]],
        mlbam_to_pid: dict[int, int],
        targets: tuple[str, ...],
        season: int,
    ) -> list[TargetCorrelation]:
        """Compute correlations for a single season."""
        # Build paired arrays for this season
        paired: dict[str, tuple[list[float], list[float]]] = {t: ([], []) for t in targets}

        for (mlbam_id, s), cand_val in candidate_values.items():
            if s != season:
                continue
            pid = mlbam_to_pid.get(mlbam_id)
            if pid is None:
                continue
            tvals = target_values.get((pid, season))
            if tvals is None:
                continue
            for t in targets:
                if t in tvals:
                    paired[t][0].append(cand_val)
                    paired[t][1].append(tvals[t])

        correlations: list[TargetCorrelation] = []
        for t in targets:
            cand_list, targ_list = paired[t]
            n = len(cand_list)
            p_r, p_p, s_r, s_p = _compute_correlation_pair(cand_list, targ_list)
            correlations.append(
                TargetCorrelation(
                    target=t,
                    pearson_r=p_r,
                    pearson_p=p_p,
                    spearman_rho=s_r,
                    spearman_p=s_p,
                    n=n,
                )
            )
        return correlations

    def _correlate_pooled(
        self,
        candidate_values: dict[tuple[int, int], float],
        target_values: dict[tuple[int, int], dict[str, float]],
        mlbam_to_pid: dict[int, int],
        targets: tuple[str, ...],
        seasons: Sequence[int],
    ) -> list[TargetCorrelation]:
        """Compute correlations pooled across all seasons."""
        paired: dict[str, tuple[list[float], list[float]]] = {t: ([], []) for t in targets}

        season_set = set(seasons)
        for (mlbam_id, s), cand_val in candidate_values.items():
            if s not in season_set:
                continue
            pid = mlbam_to_pid.get(mlbam_id)
            if pid is None:
                continue
            tvals = target_values.get((pid, s))
            if tvals is None:
                continue
            for t in targets:
                if t in tvals:
                    paired[t][0].append(cand_val)
                    paired[t][1].append(tvals[t])

        correlations: list[TargetCorrelation] = []
        for t in targets:
            cand_list, targ_list = paired[t]
            n = len(cand_list)
            p_r, p_p, s_r, s_p = _compute_correlation_pair(cand_list, targ_list)
            correlations.append(
                TargetCorrelation(
                    target=t,
                    pearson_r=p_r,
                    pearson_p=p_p,
                    spearman_rho=s_r,
                    spearman_p=s_p,
                    n=n,
                )
            )
        return correlations


def rank_columns(results: Sequence[CorrelationScanResult]) -> list[MultiColumnRanking]:
    """Rank scan results by average absolute pooled correlation, descending."""
    rankings: list[MultiColumnRanking] = []
    for result in results:
        corrs = result.pooled.correlations
        if not corrs:
            rankings.append(
                MultiColumnRanking(
                    column_spec=result.column_spec,
                    avg_abs_pearson=0.0,
                    avg_abs_spearman=0.0,
                )
            )
            continue
        avg_p = sum(abs(c.pearson_r) for c in corrs) / len(corrs)
        avg_s = sum(abs(c.spearman_rho) for c in corrs) / len(corrs)
        rankings.append(
            MultiColumnRanking(
                column_spec=result.column_spec,
                avg_abs_pearson=avg_p,
                avg_abs_spearman=avg_s,
            )
        )
    rankings.sort(key=lambda r: r.avg_abs_pearson, reverse=True)
    return rankings


class TemporalStabilityChecker:
    """Assess whether feature-target correlations are consistent across seasons."""

    def __init__(self, scanner: CorrelationScanner) -> None:
        self._scanner = scanner

    def check_temporal_stability(
        self,
        column_spec: str,
        target: str | None,
        seasons: Sequence[int],
        player_type: str,
    ) -> StabilityResult:
        """Check temporal stability of a feature's correlation with target(s).

        When *target* is a string, computes stability for that single target.
        When *target* is None, computes stability for all targets for the player type.
        """
        scan_result = self._scanner.scan_target_correlations(column_spec, seasons, player_type)

        if target is not None:
            targets_to_check = (target,)
        else:
            targets_to_check = BATTER_TARGETS if player_type == "batter" else PITCHER_TARGETS

        stabilities: list[TargetStability] = []
        for t in targets_to_check:
            stabilities.append(self._compute_stability(scan_result, t))

        return StabilityResult(
            column_spec=column_spec,
            player_type=PlayerType(player_type),
            seasons=tuple(sorted(seasons)),
            target_stabilities=tuple(stabilities),
        )

    @staticmethod
    def _compute_stability(scan_result: CorrelationScanResult, target: str) -> TargetStability:
        """Extract per-season Pearson r for a target and compute stability metrics."""
        per_season_r: list[tuple[int, float]] = []
        for season_result in scan_result.per_season:
            for corr in season_result.correlations:
                if corr.target == target:
                    per_season_r.append((season_result.season, corr.pearson_r))
                    break

        r_values = [r for _, r in per_season_r]

        if len(r_values) < 2:
            mean_r = r_values[0] if r_values else 0.0
            return TargetStability(
                target=target,
                per_season_r=tuple(per_season_r),
                mean_r=mean_r,
                std_r=0.0,
                cv=0.0,
                classification="stable",
            )

        mean_r = statistics.mean(r_values)
        std_r = statistics.stdev(r_values)

        if abs(mean_r) < 0.05:
            cv = -1.0
            classification = "stable" if std_r < 0.05 else "unstable"
        else:
            cv = std_r / abs(mean_r)
            if cv < 0.3:
                classification = "stable"
            elif cv > 0.6:
                classification = "unstable"
            else:
                classification = "moderate"

        return TargetStability(
            target=target,
            per_season_r=tuple(per_season_r),
            mean_r=mean_r,
            std_r=std_r,
            cv=cv,
            classification=classification,
        )
