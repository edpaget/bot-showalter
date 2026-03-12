"""Post-processing calibration for pitcher rate-stat predictions.

The statcast-gbm-preseason model has a systematic, non-uniform ERA/WHIP bias —
it overpredicts more for worse pitchers and for pitchers with fewer prior seasons
of data.  A linear bias correction trained on holdout seasons (2019-2023) fixes
this and makes direct rates win on both 2024 and 2025 holdouts.

Correction formula (fit on 2019-2023, tested on 2024-2025):
    corrected_era  = intercept + slope * pred_era  + prior_coef * n_prior
    corrected_whip = intercept + slope * pred_whip + prior_coef * n_prior

Where n_prior = number of seasons with ≥ min_ip IP in the lag_seasons before the
prediction season.
"""

from dataclasses import dataclass, field
from typing import Any

from fantasy_baseball_manager.repos import PitchingStatsRepo  # noqa: TC001 — used in function signatures


@dataclass(frozen=True)
class StatCalibration:
    """Coefficients for a single rate stat's linear correction."""

    intercept: float
    slope: float
    prior_coef: float


@dataclass(frozen=True)
class RateCalibrationConfig:
    """Configuration for pitcher rate-stat bias correction."""

    stats: dict[str, StatCalibration] = field(default_factory=dict)
    min_ip: float = 10.0
    lag_seasons: tuple[int, ...] = (1, 2)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RateCalibrationConfig:
        """Build config from the ``rate_calibration`` dict in model_params."""
        stats: dict[str, StatCalibration] = {}
        min_ip = float(params.get("min_ip", 10.0))
        lag_seasons = tuple(params.get("lag_seasons", (1, 2)))

        # Look for stat-specific keys like era_intercept, whip_slope, etc.
        stat_names: set[str] = set()
        for key in params:
            for suffix in ("_intercept", "_slope", "_prior_coef"):
                if key.endswith(suffix):
                    stat_names.add(key[: -len(suffix)])

        for stat in stat_names:
            intercept = params.get(f"{stat}_intercept")
            slope = params.get(f"{stat}_slope")
            prior_coef = params.get(f"{stat}_prior_coef")
            if intercept is not None and slope is not None and prior_coef is not None:
                stats[stat] = StatCalibration(
                    intercept=float(intercept),
                    slope=float(slope),
                    prior_coef=float(prior_coef),
                )

        return cls(stats=stats, min_ip=min_ip, lag_seasons=lag_seasons)


def compute_prior_season_counts(
    player_ids: set[int],
    season: int,
    pitching_stats_repo: PitchingStatsRepo,
    *,
    lag_seasons: tuple[int, ...] = (1, 2),
    min_ip: float = 10.0,
) -> dict[int, int]:
    """Count how many of the lag seasons each pitcher had ≥ min_ip IP.

    Returns a dict mapping player_id → n_prior (0 for players with no qualifying
    prior seasons).
    """
    prior_counts: dict[int, int] = {}
    # Fetch stats for each lag season once
    seasons_stats: dict[int, list[Any]] = {}
    for lag in lag_seasons:
        s = season - lag
        seasons_stats[s] = pitching_stats_repo.get_by_season(s)

    # Build per-player IP lookup for each lag season
    season_ip: dict[int, dict[int, float]] = {}  # lag_season → {player_id → max_ip}
    for s, stats_list in seasons_stats.items():
        ip_map: dict[int, float] = {}
        for stat in stats_list:
            ip = stat.ip
            if ip is not None:
                existing = ip_map.get(stat.player_id, 0.0)
                if ip > existing:
                    ip_map[stat.player_id] = ip
        season_ip[s] = ip_map

    for pid in player_ids:
        count = 0
        for lag in lag_seasons:
            s = season - lag
            ip = season_ip.get(s, {}).get(pid, 0.0)
            if ip >= min_ip:
                count += 1
        prior_counts[pid] = count

    return prior_counts


def calibrate_pitcher_rates(
    predictions: list[dict[str, Any]],
    prior_counts: dict[int, int],
    config: RateCalibrationConfig,
) -> None:
    """Apply linear bias correction to pitcher rate stats in-place.

    Only modifies predictions where player_type == "pitcher" and the stat
    is present in config.stats.
    """
    for pred in predictions:
        if pred.get("player_type") != "pitcher":
            continue
        pid = pred.get("player_id")
        n_prior = prior_counts.get(pid, 0) if pid is not None else 0
        for stat_name, cal in config.stats.items():
            raw = pred.get(stat_name)
            if raw is not None and isinstance(raw, int | float):
                pred[stat_name] = cal.intercept + cal.slope * raw + cal.prior_coef * n_prior
