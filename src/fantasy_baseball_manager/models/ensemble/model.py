"""Ensemble model — weighted-average ensemble of multiple projection systems."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.ensemble.engine import blend_rates, weighted_average, weighted_spread
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.repos.protocols import ProjectionRepo


@register("ensemble")
class EnsembleModel:
    def __init__(self, projection_repo: ProjectionRepo | None = None) -> None:
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
        assert self._projection_repo is not None, "projection_repo is required for predict"

        params = config.model_params
        components: dict[str, float] = params["components"]
        mode: str = params.get("mode", "weighted_average")
        season: int = params["season"]
        stats: Sequence[str] | None = params.get("stats")
        pt_stat: str = params.get("pt_stat", "pa")

        # Fetch projections for each component system
        system_projections: dict[str, list[Projection]] = {}
        for system in components:
            system_projections[system] = self._projection_repo.get_by_season(season, system=system)

        # Group by (player_id, player_type)
        grouped: dict[tuple[int, str], dict[str, Projection]] = defaultdict(dict)
        for system, projections in system_projections.items():
            for proj in projections:
                grouped[(proj.player_id, proj.player_type)][system] = proj

        # Compute ensemble for each player
        predictions: list[dict[str, Any]] = []
        all_distributions: list[dict[str, Any]] = []
        for (player_id, player_type), system_map in grouped.items():
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
                # Use union of all stats across available systems
                all_keys: set[str] = set()
                for stat_json, _ in pairs:
                    all_keys.update(stat_json.keys())
                effective_stats = sorted(all_keys)

            # Apply engine function
            if mode == "blend_rates":
                result_stats = blend_rates(pairs, rate_stats=list(effective_stats), pt_stat=pt_stat)
            else:
                result_stats = weighted_average(pairs, stats=effective_stats)

            if result_stats:
                predictions.append(
                    {
                        "player_id": player_id,
                        "season": season,
                        "player_type": player_type,
                        **result_stats,
                        "_components": dict(components),
                        "_mode": mode,
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
                            "season": season,
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
