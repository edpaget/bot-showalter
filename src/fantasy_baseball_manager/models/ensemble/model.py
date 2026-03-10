"""Ensemble model — weighted-average ensemble of multiple projection systems."""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import (
    ArtifactType,
    ConsensusLookup,
    resolve_playing_time,
)
from fantasy_baseball_manager.models.ensemble.engine import (
    blend_rates,
    per_stat_weighted,
    routed,
    weighted_average,
    weighted_spread,
)
from fantasy_baseball_manager.models.ensemble.stat_groups import expand_route_groups, validate_coverage
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.repos import (
    ProjectionRepo,  # noqa: TC001 — used in __init__ signature evaluated by inspect.signature()
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.domain import Projection


logger = logging.getLogger(__name__)


@register("ensemble")
class EnsembleModel:
    def __init__(self, projection_repo: ProjectionRepo) -> None:
        self._projection_repo = projection_repo

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def description(self) -> str:
        return "Weighted-average ensemble of multiple projection systems"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def predict(self, config: ModelConfig) -> PredictResult:
        params = config.model_params
        components: dict[str, float] = params["components"]
        mode: str = params.get("mode", "weighted_average")
        seasons: list[int] = config.seasons
        stats: Sequence[str] | None = params.get("stats")
        pt_stat: str = params.get("pt_stat", "pa")
        versions: dict[str, str] = params.get("versions", {})
        pt_mode: str = params.get("playing_time", "native")

        # Expand route_groups into per-stat routes
        route_groups_param: dict[str, str] | None = params.get("route_groups")
        if route_groups_param is not None:
            expanded_routes = expand_route_groups(
                route_groups=route_groups_param,
                routes=params.get("routes"),
                custom_groups=params.get("stat_groups"),
                league=params.get("league"),
            )
            params["routes"] = expanded_routes
            mode = "routed"

        # Coverage validation when league is present and mode is routed
        league = params.get("league")
        if league is not None and mode == "routed":
            uncovered = validate_coverage(params["routes"], league)
            for stat in uncovered:
                logger.warning("Uncovered league-required stat: %s", stat)
            if uncovered and params.get("check"):
                msg = f"Uncovered league-required stats: {', '.join(uncovered)}"
                raise ValueError(msg)

        # Dry run: return metadata without fetching projections
        if params.get("dry_run"):
            output_path = config.output_dir or config.artifacts_dir
            meta: dict[str, Any] = {
                "_components": dict(components),
                "_mode": mode,
            }
            if mode == "routed":
                meta["_routes"] = dict(params["routes"])
            return PredictResult(
                model_name="ensemble",
                predictions=[meta],
                output_path=output_path,
            )

        # Fetch projections for each component system, filtered to target seasons
        system_projections: dict[str, list[Projection]] = {}
        for system in components:
            if system in versions:
                all_versions = self._projection_repo.get_by_system_version(system, versions[system])
                system_projections[system] = [p for p in all_versions if p.season in seasons]
            else:
                projs: list[Projection] = []
                for s in seasons:
                    projs.extend(self._projection_repo.get_by_season(s, system=system))
                system_projections[system] = projs

        # Group by (player_id, player_type, season)
        grouped: dict[tuple[int, str, int], dict[str, Projection]] = defaultdict(dict)
        for system, projections in system_projections.items():
            for proj in projections:
                grouped[(proj.player_id, proj.player_type, proj.season)][system] = proj

        # Build consensus PT lookup per season when requested
        consensus_by_season: dict[int, ConsensusLookup | None] = {}
        for s in seasons:
            consensus_by_season[s] = resolve_playing_time(
                pt_mode,
                s,
                fetch_projections=self._projection_repo.get_by_season,
            )

        # Compute ensemble for each player-season
        predictions: list[dict[str, Any]] = []
        all_distributions: list[dict[str, Any]] = []
        for (player_id, player_type, pred_season), system_map in grouped.items():
            # Collect (stat_json, weight) pairs for available systems
            pairs: list[tuple[dict[str, Any], float]] = []
            for system, weight in components.items():
                if system in system_map:
                    pairs.append((system_map[system].stat_json, weight))

            if not pairs:
                continue

            # Determine stats to average
            effective_stats = stats
            if effective_stats is None:
                # Use union of all stats across available systems (skip metadata keys)
                all_keys: set[str] = set()
                for stat_json, _ in pairs:
                    all_keys.update(k for k in stat_json if not k.startswith("_"))
                effective_stats = sorted(all_keys)

            # Resolve consensus PT for this player
            consensus = consensus_by_season.get(pred_season)
            cpt: float | None = None
            cpt_key: str = pt_stat
            if consensus is not None:
                cpt_key = "ip" if player_type == "pitcher" else "pa"
                cpt = (consensus.pitching_pt if player_type == "pitcher" else consensus.batting_pt).get(player_id)

            # Apply engine function
            param_stat_weights: dict[str, dict[str, float]] | None = params.get("stat_weights")
            if mode == "routed":
                routes: dict[str, str] = params["routes"]
                fallback: str | None = params.get("fallback")
                sys_stats = {sys: proj.stat_json for sys, proj in system_map.items()}
                result_stats = routed(sys_stats, routes, fallback)
                extra_meta: dict[str, Any] = {"_routes": dict(routes)}
            elif param_stat_weights is not None:
                sys_stats = {sys: proj.stat_json for sys, proj in system_map.items()}
                result_stats = per_stat_weighted(sys_stats, param_stat_weights)
                extra_meta = {"_stat_weights": param_stat_weights}
            elif mode == "blend_rates":
                result_stats = blend_rates(pairs, rate_stats=list(effective_stats), pt_stat=cpt_key, consensus_pt=cpt)
                extra_meta = {}
            else:
                result_stats = weighted_average(pairs, stats=effective_stats)
                if cpt is not None and result_stats:
                    result_stats[cpt_key] = cpt
                extra_meta = {}

            if result_stats:
                predictions.append(
                    {
                        "player_id": player_id,
                        "season": pred_season,
                        "player_type": player_type,
                        **result_stats,
                        "_components": dict(components),
                        "_mode": mode,
                        **extra_meta,
                    }
                )

            # Compute distributional spread when ≥2 systems contribute
            if len(pairs) >= 2:
                spread = weighted_spread(pairs, stats=effective_stats)
                for stat_name, dist in spread.items():
                    all_distributions.append(
                        {
                            "player_id": player_id,
                            "player_type": player_type,
                            "season": pred_season,
                            "stat": stat_name,
                            "p10": dist.p10,
                            "p25": dist.p25,
                            "p50": dist.p50,
                            "p75": dist.p75,
                            "p90": dist.p90,
                            "mean": dist.mean,
                            "std": dist.std,
                        }
                    )

        output_path = config.output_dir or config.artifacts_dir
        return PredictResult(
            model_name="ensemble",
            predictions=predictions,
            output_path=output_path,
            distributions=all_distributions if all_distributions else None,
        )
