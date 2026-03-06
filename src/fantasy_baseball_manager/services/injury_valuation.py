from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import InjuryValueDelta, PlayerValuation, discount_projections
from fantasy_baseball_manager.models import ModelConfig

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Predictable
    from fantasy_baseball_manager.repos import PlayerRepo, ProjectionRepo, ValuationRepo
    from fantasy_baseball_manager.services.injury_profiler import InjuryProfiler


def _run_injury_adjusted_predictions(
    season: int,
    league: Any,
    projection_system: str,
    projection_version: str | None,
    season_list: list[int],
    profiler: InjuryProfiler,
    model: Predictable,
    player_repo: PlayerRepo,
    projection_repo: ProjectionRepo,
) -> tuple[list[dict[str, Any]], dict[int, float], dict[int, str]]:
    """Shared core: compute injury-adjusted predictions, injury_map, and player names."""
    estimates = profiler.list_games_lost_estimates(season_list, projection_season=season)
    injury_map = {est.player_id: est.expected_days_lost for est, _, _ in estimates}

    # Read projections and apply injury discounts before passing to model
    if projection_version is not None:
        projections = projection_repo.get_by_system_version(projection_system, projection_version)
        projections = [p for p in projections if p.season == season]
    else:
        projections = projection_repo.get_by_season(season, system=projection_system)
    projections = discount_projections(projections, injury_map)

    config = ModelConfig(
        seasons=[season],
        model_params={
            "league": league,
            "projection_system": projection_system,
            "projection_version": projection_version,
            "projections": projections,
        },
        version="injury-adjusted",
    )
    result = model.predict(config)

    player_ids = {p["player_id"] for p in result.predictions}
    players = player_repo.get_by_ids(list(player_ids))
    player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

    return result.predictions, injury_map, player_names


def compute_injury_adjusted_valuations_list(
    season: int,
    league: Any,
    projection_system: str,
    projection_version: str | None,
    season_list: list[int],
    profiler: InjuryProfiler,
    model: Predictable,
    player_repo: PlayerRepo,
    projection_repo: ProjectionRepo,
) -> list[PlayerValuation]:
    """Compute injury-adjusted valuations and return as PlayerValuation objects."""
    predictions, _, player_names = _run_injury_adjusted_predictions(
        season,
        league,
        projection_system,
        projection_version,
        season_list,
        profiler,
        model,
        player_repo,
        projection_repo,
    )

    valuations: list[PlayerValuation] = []
    for pred in predictions:
        name = player_names.get(pred["player_id"], f"Player {pred['player_id']}")
        valuations.append(
            PlayerValuation(
                player_name=name,
                system="zar",
                version="injury-adjusted",
                projection_system=projection_system,
                projection_version=projection_version or "",
                player_type=pred.get("player_type", ""),
                position=pred.get("position", ""),
                value=pred["value"],
                rank=pred["rank"],
                category_scores=pred.get("category_scores", {}),
            )
        )
    return valuations


def compute_injury_adjusted_deltas(
    season: int,
    league: Any,
    projection_system: str,
    projection_version: str | None,
    season_list: list[int],
    profiler: InjuryProfiler,
    model: Predictable,
    player_repo: PlayerRepo,
    valuation_repo: ValuationRepo,
    projection_repo: ProjectionRepo,
    system: str = "zar",
    version: str = "1.0",
) -> list[InjuryValueDelta]:
    """Compute deltas between original and injury-adjusted valuations."""
    original_vals = valuation_repo.get_by_season(season, system=system)
    original_vals = [v for v in original_vals if v.version == version]
    if not original_vals:
        return []

    orig_by_player = {v.player_id: v for v in original_vals}

    predictions, injury_map, player_names = _run_injury_adjusted_predictions(
        season,
        league,
        projection_system,
        projection_version,
        season_list,
        profiler,
        model,
        player_repo,
        projection_repo,
    )
    adj_by_player = {p["player_id"]: p for p in predictions}

    deltas: list[InjuryValueDelta] = []
    for pid in orig_by_player:
        adj = adj_by_player.get(pid)
        if adj is None:
            continue
        orig_val = orig_by_player[pid]
        delta_val = adj["value"] - orig_val.value
        deltas.append(
            InjuryValueDelta(
                player_name=player_names.get(pid, f"Player {pid}"),
                original_value=orig_val.value,
                adjusted_value=adj["value"],
                value_delta=delta_val,
                original_rank=orig_val.rank,
                adjusted_rank=adj["rank"],
                rank_change=orig_val.rank - adj["rank"],
                expected_days_lost=injury_map.get(pid, 0.0),
            )
        )

    deltas.sort(key=lambda d: d.value_delta)
    return deltas
