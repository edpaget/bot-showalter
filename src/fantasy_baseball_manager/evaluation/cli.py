from collections.abc import Callable
from typing import Annotated

import typer

from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.evaluation.harness import EvaluationConfig, evaluate
from fantasy_baseball_manager.evaluation.models import RankAccuracy, SourceEvaluation, StatAccuracy
from fantasy_baseball_manager.marcel.batting import project_batters
from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource, StatsDataSource
from fantasy_baseball_manager.marcel.pitching import project_pitchers
from fantasy_baseball_manager.valuation.models import StatCategory
from fantasy_baseball_manager.valuation.projection_source import ProjectionSource, SimpleProjectionSource

_DEFAULT_BATTING: tuple[StatCategory, ...] = (StatCategory.HR, StatCategory.SB, StatCategory.OBP)
_DEFAULT_PITCHING: tuple[StatCategory, ...] = (StatCategory.K, StatCategory.ERA, StatCategory.WHIP)

# Module-level factory for dependency injection in tests
_data_source_factory: Callable[[], StatsDataSource] = PybaseballDataSource


def set_data_source_factory(factory: Callable[[], StatsDataSource]) -> None:
    global _data_source_factory
    _data_source_factory = factory


def _format_stat_accuracy_table(label: str, accuracies: tuple[StatAccuracy, ...]) -> str:
    if not accuracies:
        return f"  {label}: no matched players"
    lines: list[str] = []
    lines.append(f"  {label} stat accuracy (n={accuracies[0].sample_size}):")
    lines.append(f"    {'Category':<10} {'RMSE':>8} {'MAE':>8} {'Corr':>8}")
    lines.append(f"    {'-' * 38}")
    for sa in accuracies:
        lines.append(f"    {sa.category.value:<10} {sa.rmse:>8.3f} {sa.mae:>8.3f} {sa.correlation:>8.3f}")
    return "\n".join(lines)


def _format_rank_accuracy(label: str, ra: RankAccuracy | None) -> str:
    if ra is None:
        return f"  {label} rank accuracy: insufficient data"
    return (
        f"  {label} rank accuracy (n={ra.sample_size}):\n"
        f"    Spearman rho: {ra.spearman_rho:.3f}\n"
        f"    Top-{ra.top_n} precision: {ra.top_n_precision:.3f}"
    )


def _format_evaluation(se: SourceEvaluation) -> str:
    lines: list[str] = []
    lines.append(f"Source: {se.source_name} ({se.year})")
    lines.append(_format_stat_accuracy_table("Batting", se.batting_stat_accuracy))
    lines.append(_format_rank_accuracy("Batting", se.batting_rank_accuracy))
    lines.append(_format_stat_accuracy_table("Pitching", se.pitching_stat_accuracy))
    lines.append(_format_rank_accuracy("Pitching", se.pitching_rank_accuracy))
    return "\n".join(lines)


def _build_source(engine: str, data_source: StatsDataSource, year: int) -> tuple[str, ProjectionSource]:
    if engine == "marcel":
        batting_proj = project_batters(data_source, year)
        pitching_proj = project_pitchers(data_source, year)
        return (engine, SimpleProjectionSource(_batting=batting_proj, _pitching=pitching_proj))
    raise ValueError(f"Unknown engine: {engine}")


def evaluate_cmd(
    year: Annotated[int, typer.Argument(help="Season year to evaluate against.")],
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    min_pa: Annotated[int, typer.Option(help="Minimum plate appearances for inclusion.")] = 200,
    min_ip: Annotated[float, typer.Option(help="Minimum innings pitched for inclusion.")] = 50.0,
    top_n: Annotated[int, typer.Option(help="N for top-N precision metric.")] = 20,
) -> None:
    """Evaluate projection sources against actual season outcomes."""
    validate_engine(engine)

    data_source = _data_source_factory()

    config = EvaluationConfig(
        year=year,
        batting_categories=_DEFAULT_BATTING,
        pitching_categories=_DEFAULT_PITCHING,
        min_pa=min_pa,
        min_ip=min_ip,
        top_n=top_n,
    )

    source = _build_source(engine, data_source, year)
    result = evaluate(sources=[source], data_source=data_source, config=config)

    for i, se in enumerate(result.evaluations):
        if i > 0:
            typer.echo()
        typer.echo(_format_evaluation(se))
