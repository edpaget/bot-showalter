from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from fantasy_baseball_manager.domain import (
    ArtifactType,
    EligibilityProvider,
    GamesLostEstimator,
    Position,
    Valuation,
    compute_age,
)
from fantasy_baseball_manager.models.playing_time.engine import ResidualBuckets, _bucket_key
from fantasy_baseball_manager.models.playing_time.serialization import load_residual_buckets
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.zar.engine import ZarPipelineResult, compute_budget_split
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.models.zar.positions import best_position, build_roster_spots
from fantasy_baseball_manager.repos import (  # noqa: TC001 — used in __init__ signature evaluated by inspect.signature()
    PlayerRepo,
    PositionAppearanceRepo,
    ProjectionRepo,
    ValuationRepo,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CategoryConfig, LeagueSettings, Projection


class ScenarioGenerator(Protocol):
    """Generate weighted PT scenario projections for a pool of players."""

    def __call__(
        self,
        projections: list[Projection],
        residual_buckets_map: dict[str, ResidualBuckets],
        player_bucket_keys: dict[int, str],
        scenario_weights: dict[int, float] | None = None,
    ) -> dict[int, list[tuple[Projection, float]]]: ...


class DistributionalZarRunner(Protocol):
    """Run ZAR at each scenario level and compute expected dollar values."""

    def __call__(
        self,
        point_stats: list[dict[str, float]],
        scenario_stats: list[list[tuple[dict[str, float], float]]],
        categories: list[CategoryConfig],
        player_positions: list[list[str]],
        roster_spots: dict[str, int],
        num_teams: int,
        budget: float,
        *,
        stdev_overrides: dict[str, float] | None = None,
    ) -> ZarPipelineResult: ...


def _extract_stats(projections: list[Projection]) -> list[dict[str, float]]:
    """Convert projection stat_json dicts to float-valued dicts."""
    result: list[dict[str, float]] = []
    for proj in projections:
        row: dict[str, float] = {}
        for k, v in proj.stat_json.items():
            if isinstance(v, int | float):
                row[k] = float(v)
        result.append(row)
    return result


@register("zar-distributional")
class ZarDistributionalModel:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        position_repo: PositionAppearanceRepo,
        player_repo: PlayerRepo | None = None,
        valuation_repo: ValuationRepo | None = None,
        eligibility_service: EligibilityProvider | None = None,
        injury_profiler: GamesLostEstimator | None = None,
        scenario_generator: ScenarioGenerator | None = None,
        distributional_zar_runner: DistributionalZarRunner | None = None,
    ) -> None:
        self._zar = ZarModel(
            projection_repo,
            position_repo,
            player_repo,
            valuation_repo,
            eligibility_service,
        )
        self._profiler = injury_profiler
        self._player_repo = player_repo
        self._projection_repo = projection_repo
        self._position_repo = position_repo
        self._valuation_repo = valuation_repo
        self._eligibility_service = eligibility_service
        self._generate_scenarios = scenario_generator
        self._run_distributional_zar = distributional_zar_runner

    @property
    def name(self) -> str:
        return "zar-distributional"

    @property
    def description(self) -> str:
        return "Distributional ZAR valuations using playing-time scenario weighting"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def predict(self, config: ModelConfig) -> PredictResult:
        if self._eligibility_service is None:
            msg = "eligibility_service is required for predict()"
            raise TypeError(msg)
        if self._generate_scenarios is None or self._run_distributional_zar is None:
            msg = "scenario_generator and distributional_zar_runner are required for predict()"
            raise TypeError(msg)

        league: LeagueSettings = config.model_params["league"]
        proj_system: str = config.model_params["projection_system"]
        proj_version: str | None = config.model_params.get("projection_version")
        season = config.seasons[0] if config.seasons else 2025
        version = config.version or "1.0"

        # 1. Load residual buckets (override via model_params or from artifacts)
        residual_buckets: dict[str, ResidualBuckets] | None = config.model_params.get("_residual_buckets")
        if residual_buckets is None:
            buckets_path = Path(config.artifacts_dir or "") / "playing-time-model" / "pt_residual_buckets.joblib"
            if buckets_path.exists():
                residual_buckets = load_residual_buckets(buckets_path)

        # 2. If no residual buckets available, fall back to point-estimate ZAR
        if residual_buckets is None:
            zar_params = dict(config.model_params)
            zar_params["valuation_system"] = "zar-distributional"
            zar_config = dataclasses.replace(config, model_params=zar_params)
            result = self._zar.predict(zar_config)
            return PredictResult(
                model_name=self.name,
                predictions=result.predictions,
                output_path=result.output_path,
            )

        # 3. Read projections
        projections: list[Projection] = config.model_params.get("projections") or []
        if not projections:
            if proj_version is not None:
                projections = self._projection_repo.get_by_system_version(proj_system, proj_version)
                projections = [p for p in projections if p.season == season]
            else:
                projections = self._projection_repo.get_by_season(season, system=proj_system)

        # 4. Build bucket keys for each player
        player_bucket_keys = self._build_bucket_keys(projections, season, config)

        # 5. Generate scenarios
        scenarios = self._generate_scenarios(projections, residual_buckets, player_bucket_keys, None)

        # 6. Split into batters/pitchers
        min_pa = league.eligibility.min_pa or 1
        min_ip = league.eligibility.min_ip or 1
        batter_projs = [p for p in projections if p.player_type == "batter" and p.stat_json.get("pa", 0) >= min_pa]
        pitcher_projs = [p for p in projections if p.player_type == "pitcher" and p.stat_json.get("ip", 0) >= min_ip]

        # 7. Budget split
        batter_budget, pitcher_budget = compute_budget_split(league)

        # 8. Position maps
        position_map = self._eligibility_service.get_batter_positions(season, league)

        # 9. Value batters
        batter_valuations = self._value_pool_distributional(
            batter_projs,
            scenarios,
            list(league.batting_categories),
            position_map,
            league,
            batter_budget,
            "batter",
            season,
            version,
            proj_system,
        )

        # 10. Value pitchers
        pitcher_position_map = self._eligibility_service.get_pitcher_positions(
            season,
            league,
            [p.player_id for p in pitcher_projs],
        )
        if league.pitcher_positions:
            pitcher_roster_spots = dict(league.pitcher_positions)
        else:
            pitcher_roster_spots = {Position.P: league.roster_pitchers}
        pitcher_valuations = self._value_pool_distributional(
            pitcher_projs,
            scenarios,
            list(league.pitching_categories),
            pitcher_position_map,
            league,
            pitcher_budget,
            "pitcher",
            season,
            version,
            proj_system,
            pitcher_roster_spots=pitcher_roster_spots,
        )

        # 11. Rank all valuations combined
        all_valuations = batter_valuations + pitcher_valuations
        all_valuations.sort(key=lambda v: v.value, reverse=True)
        ranked = [dataclasses.replace(v, rank=i) for i, v in enumerate(all_valuations, 1)]

        # 12. Persist
        if self._valuation_repo is not None:
            for v in ranked:
                self._valuation_repo.upsert(v)

        # 13. Build PredictResult
        predictions: list[dict[str, Any]] = []
        for v in ranked:
            predictions.append(
                {
                    "player_id": v.player_id,
                    "season": v.season,
                    "player_type": v.player_type,
                    "position": v.position,
                    "value": v.value,
                    "rank": v.rank,
                    "category_scores": v.category_scores,
                }
            )

        return PredictResult(
            model_name=self.name,
            predictions=predictions,
            output_path=config.output_dir or config.artifacts_dir,
        )

    def _build_bucket_keys(
        self,
        projections: list[Projection],
        season: int,
        config: ModelConfig,
    ) -> dict[int, str]:
        """Build bucket keys (age + injury status) for each player."""
        player_ids = [p.player_id for p in projections]

        # Get ages from player repo
        ages: dict[int, int | None] = {}
        if self._player_repo is not None:
            players = self._player_repo.get_by_ids(player_ids)
            for player in players:
                if player.id is not None:
                    ages[player.id] = compute_age(player.birth_date, season)

        # Get IL history from injury profiler
        il_days: dict[int, float] = {}
        if self._profiler is not None:
            seasons_back: int = config.model_params.get("seasons_back", 5)
            season_list = list(range(season - seasons_back, season))
            estimates = self._profiler.list_games_lost_estimates(season_list, projection_season=season)
            il_days = {est.player_id: est.expected_days_lost for est, _, _ in estimates}

        # Build bucket key for each player
        result: dict[int, str] = {}
        for pid in player_ids:
            age = ages.get(pid)
            days = il_days.get(pid)
            result[pid] = _bucket_key(age, days)
        return result

    def _value_pool_distributional(
        self,
        projections: list[Projection],
        scenarios: dict[int, list[tuple[Projection, float]]],
        categories: list[Any],
        position_map: dict[int, list[str]],
        league: LeagueSettings,
        budget: float,
        player_type: str,
        season: int,
        version: str,
        proj_system: str,
        *,
        pitcher_roster_spots: dict[str, int] | None = None,
    ) -> list[Valuation]:
        """Run distributional ZAR for one player pool."""
        if not projections:
            return []

        # Build position lists
        if pitcher_roster_spots is not None and "P" in pitcher_roster_spots:
            no_pos: list[str] = ["P"]
        elif league.roster_util > 0:
            no_pos = ["UTIL"]
        else:
            no_pos = []
        player_positions = [position_map.get(p.player_id, no_pos) for p in projections]
        roster_spots = build_roster_spots(league, pitcher_roster_spots=pitcher_roster_spots)

        # Point-estimate stats
        point_stats = _extract_stats(projections)

        # Build scenario stats: list of (stats_dict, weight) per player
        scenario_stats: list[list[tuple[dict[str, float], float]]] = []
        for proj in projections:
            player_scenarios = scenarios.get(proj.player_id)
            if player_scenarios is None:
                # Single scenario at point estimate
                scenario_stats.append([(_extract_stats([proj])[0], 1.0)])
            else:
                entries: list[tuple[dict[str, float], float]] = []
                for scenario_proj, weight in player_scenarios:
                    stats = _extract_stats([scenario_proj])[0]
                    entries.append((stats, weight))
                scenario_stats.append(entries)

        run_zar = self._run_distributional_zar
        if run_zar is None:  # pragma: no cover — guarded by predict()
            msg = "distributional_zar_runner is required"
            raise TypeError(msg)
        result = run_zar(
            point_stats,
            scenario_stats,
            categories,
            player_positions,
            roster_spots,
            league.teams,
            budget,
        )

        # Build Valuation objects
        valuations: list[Valuation] = []
        for i, proj in enumerate(projections):
            best_pos = best_position(player_positions[i], result.replacement)
            valuations.append(
                Valuation(
                    player_id=proj.player_id,
                    season=season,
                    system="zar-distributional",
                    version=version,
                    projection_system=proj_system,
                    projection_version=proj.version,
                    player_type=player_type,
                    position=best_pos,
                    value=round(result.dollar_values[i], 2),
                    rank=0,
                    category_scores=dict(result.z_scores[i].category_z),
                )
            )
        return valuations
