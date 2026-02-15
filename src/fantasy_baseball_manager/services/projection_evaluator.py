from typing import TypeVar

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    SystemMetrics,
    compute_stat_metrics,
)
from fantasy_baseball_manager.domain.pitching_stats import PitchingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.projection_accuracy import (
    ProjectionComparison,
    compare_to_batting_actuals,
    compare_to_pitching_actuals,
    missing_batting_comparisons,
    missing_pitching_comparisons,
)
from fantasy_baseball_manager.repos.protocols import (
    BattingStatsRepo,
    PitchingStatsRepo,
    ProjectionRepo,
)

_T = TypeVar("_T", BattingStats, PitchingStats)


def _top_by_war(items: list[_T], top: int) -> list[_T]:
    return sorted(items, key=lambda x: x.war if x.war is not None else 0.0, reverse=True)[:top]


class ProjectionEvaluator:
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
        season: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> SystemMetrics:
        projections = self._projection_repo.get_by_system_version(system, version)
        projections = [p for p in projections if p.season == season]

        source_type = "first_party"
        if projections:
            source_type = projections[0].source_type

        batter_projs: dict[int, Projection] = {}
        pitcher_projs: dict[int, Projection] = {}
        for proj in projections:
            if proj.player_type == "batter":
                batter_projs[proj.player_id] = proj
            elif proj.player_type == "pitcher":
                pitcher_projs[proj.player_id] = proj

        comparisons: list[ProjectionComparison] = []

        # Iterate over actuals (not projections)
        batting_actuals = self._batting_repo.get_by_season(season, source=actuals_source)
        if top is not None:
            batting_actuals = _top_by_war(batting_actuals, top)
        for actual in batting_actuals:
            proj = batter_projs.get(actual.player_id)
            if proj is not None:
                comparisons.extend(compare_to_batting_actuals(proj, actual))
            else:
                comparisons.extend(missing_batting_comparisons(actual))

        pitching_actuals = self._pitching_repo.get_by_season(season, source=actuals_source)
        if top is not None:
            pitching_actuals = _top_by_war(pitching_actuals, top)
        for actual in pitching_actuals:
            proj = pitcher_projs.get(actual.player_id)
            if proj is not None:
                comparisons.extend(compare_to_pitching_actuals(proj, actual))
            else:
                comparisons.extend(missing_pitching_comparisons(actual))

        metrics = compute_stat_metrics(comparisons, stats)

        return SystemMetrics(
            system=system,
            version=version,
            source_type=source_type,
            metrics=metrics,
        )

    def compare(
        self,
        systems: list[tuple[str, str]],
        season: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> ComparisonResult:
        system_metrics: list[SystemMetrics] = []
        all_stat_names: set[str] = set()

        for system, version in systems:
            metrics = self.evaluate(system, version, season, stats, actuals_source, top=top)
            system_metrics.append(metrics)
            all_stat_names.update(metrics.metrics.keys())

        return ComparisonResult(
            season=season,
            stats=sorted(all_stat_names),
            systems=system_metrics,
        )
