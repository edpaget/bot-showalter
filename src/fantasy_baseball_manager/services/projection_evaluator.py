from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    SystemMetrics,
    compute_stat_metrics,
)
from fantasy_baseball_manager.domain.projection_accuracy import (
    ProjectionComparison,
    compare_to_batting_actuals,
    compare_to_pitching_actuals,
)
from fantasy_baseball_manager.repos.protocols import (
    BattingStatsRepo,
    PitchingStatsRepo,
    ProjectionRepo,
)


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
    ) -> SystemMetrics:
        projections = self._projection_repo.get_by_system_version(system, version)
        projections = [p for p in projections if p.season == season]

        comparisons: list[ProjectionComparison] = []
        source_type = "first_party"

        if projections:
            source_type = projections[0].source_type

        for proj in projections:
            if proj.player_type == "batter":
                actuals = self._batting_repo.get_by_player_season(
                    proj.player_id,
                    season,
                    source=actuals_source,
                )
                if actuals:
                    comparisons.extend(compare_to_batting_actuals(proj, actuals[0]))
            elif proj.player_type == "pitcher":
                actuals = self._pitching_repo.get_by_player_season(
                    proj.player_id,
                    season,
                    source=actuals_source,
                )
                if actuals:
                    comparisons.extend(compare_to_pitching_actuals(proj, actuals[0]))

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
    ) -> ComparisonResult:
        system_metrics: list[SystemMetrics] = []
        all_stat_names: set[str] = set()

        for system, version in systems:
            metrics = self.evaluate(system, version, season, stats, actuals_source)
            system_metrics.append(metrics)
            all_stat_names.update(metrics.metrics.keys())

        return ComparisonResult(
            season=season,
            stats=sorted(all_stat_names),
            systems=system_metrics,
        )
