from typing import TypeVar

from fantasy_baseball_manager.domain.batting_stats import BattingStats
from collections import defaultdict

from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    StratifiedComparisonResult,
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

        batting_actuals = self._batting_repo.get_by_season(season, source=actuals_source)
        if top is not None:
            batting_actuals = _top_by_war(batting_actuals, top)

        pitching_actuals = self._pitching_repo.get_by_season(season, source=actuals_source)
        if top is not None:
            pitching_actuals = _top_by_war(pitching_actuals, top)

        comparisons = self._build_comparisons(
            batter_projs,
            pitcher_projs,
            batting_actuals,
            pitching_actuals,
        )

        metrics = compute_stat_metrics(comparisons, stats)

        return SystemMetrics(
            system=system,
            version=version,
            source_type=source_type,
            metrics=metrics,
        )

    def _build_comparisons(
        self,
        batter_projs: dict[int, Projection],
        pitcher_projs: dict[int, Projection],
        batting_actuals: list[BattingStats],
        pitching_actuals: list[PitchingStats],
    ) -> list[ProjectionComparison]:
        comparisons: list[ProjectionComparison] = []
        for actual in batting_actuals:
            proj = batter_projs.get(actual.player_id)
            if proj is not None:
                comparisons.extend(compare_to_batting_actuals(proj, actual))
            else:
                comparisons.extend(missing_batting_comparisons(actual))
        for actual in pitching_actuals:
            proj = pitcher_projs.get(actual.player_id)
            if proj is not None:
                comparisons.extend(compare_to_pitching_actuals(proj, actual))
            else:
                comparisons.extend(missing_pitching_comparisons(actual))
        return comparisons

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

    def evaluate_stratified(
        self,
        system: str,
        version: str,
        season: int,
        cohort_assignments: dict[int, str],
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> dict[str, SystemMetrics]:
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

        batting_actuals = self._batting_repo.get_by_season(season, source=actuals_source)
        if top is not None:
            batting_actuals = _top_by_war(batting_actuals, top)

        pitching_actuals = self._pitching_repo.get_by_season(season, source=actuals_source)
        if top is not None:
            pitching_actuals = _top_by_war(pitching_actuals, top)

        # Partition actuals by cohort
        batting_by_cohort: dict[str, list[BattingStats]] = defaultdict(list)
        for actual in batting_actuals:
            label = cohort_assignments.get(actual.player_id)
            if label is not None:
                batting_by_cohort[label].append(actual)

        pitching_by_cohort: dict[str, list[PitchingStats]] = defaultdict(list)
        for actual in pitching_actuals:
            label = cohort_assignments.get(actual.player_id)
            if label is not None:
                pitching_by_cohort[label].append(actual)

        all_labels = set(batting_by_cohort.keys()) | set(pitching_by_cohort.keys())
        result: dict[str, SystemMetrics] = {}
        for label in sorted(all_labels):
            comparisons = self._build_comparisons(
                batter_projs,
                pitcher_projs,
                batting_by_cohort.get(label, []),
                pitching_by_cohort.get(label, []),
            )
            metrics = compute_stat_metrics(comparisons, stats)
            result[label] = SystemMetrics(
                system=system,
                version=version,
                source_type=source_type,
                metrics=metrics,
            )
        return result

    def compare_stratified(
        self,
        systems: list[tuple[str, str]],
        season: int,
        cohort_assignments: dict[int, str],
        dimension: str,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> StratifiedComparisonResult:
        # Collect per-cohort metrics for each system
        all_cohort_labels: set[str] = set()
        per_system: list[dict[str, SystemMetrics]] = []

        for system, version in systems:
            cohort_metrics = self.evaluate_stratified(
                system,
                version,
                season,
                cohort_assignments,
                stats,
                actuals_source,
                top=top,
            )
            per_system.append(cohort_metrics)
            all_cohort_labels.update(cohort_metrics.keys())

        # Build ComparisonResult per cohort
        all_stat_names: set[str] = set()
        cohorts: dict[str, ComparisonResult] = {}
        for label in sorted(all_cohort_labels):
            system_metrics_list: list[SystemMetrics] = []
            for sys_metrics in per_system:
                sm = sys_metrics.get(label)
                if sm is not None:
                    system_metrics_list.append(sm)
                    all_stat_names.update(sm.metrics.keys())
            cohorts[label] = ComparisonResult(
                season=season,
                stats=sorted({s for sm in system_metrics_list for s in sm.metrics}),
                systems=system_metrics_list,
            )

        return StratifiedComparisonResult(
            dimension=dimension,
            season=season,
            stats=sorted(all_stat_names),
            cohorts=cohorts,
        )
