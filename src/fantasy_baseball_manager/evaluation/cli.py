import json
from typing import Annotated

import typer

from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.evaluation.harness import (
    EvaluationConfig,
    StratificationConfig,
    compare_sources,
    evaluate,
)
from fantasy_baseball_manager.evaluation.models import (
    HeadToHeadResult,
    RankAccuracy,
    SourceEvaluation,
    StatAccuracy,
    StratumAccuracy,
)
from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.pipeline.source import PipelineProjectionSource
from fantasy_baseball_manager.services import get_container, set_container
from fantasy_baseball_manager.valuation.projection_source import ProjectionSource

__all__ = ["evaluate_cmd", "set_container"]


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


def _format_strata(strata: tuple[StratumAccuracy, ...], label: str) -> str:
    if not strata:
        return ""
    lines = [f"  {label} by segment:"]
    lines.append(f"    {'Segment':<15} {'N':>5} {'RMSE':>8} {'Corr':>8}")
    lines.append(f"    {'-' * 40}")
    for s in strata:
        if s.stat_accuracy:
            avg_rmse = sum(sa.rmse for sa in s.stat_accuracy) / len(s.stat_accuracy)
            avg_corr = sum(sa.correlation for sa in s.stat_accuracy) / len(s.stat_accuracy)
            lines.append(f"    {s.stratum_name:<15} {s.sample_size:>5} {avg_rmse:>8.3f} {avg_corr:>8.3f}")
    return "\n".join(lines)


def _format_head_to_head(results: tuple[HeadToHeadResult, ...]) -> str:
    if not results:
        return ""
    lines: list[str] = []
    for h2h in results:
        pct_a = h2h.a_win_pct * 100
        pct_b = (h2h.b_wins / h2h.sample_size * 100) if h2h.sample_size > 0 else 0.0
        lines.append(f"=== Head-to-Head: {h2h.source_a} vs {h2h.source_b} ({h2h.category.value}) ===")
        lines.append(
            f"  N={h2h.sample_size}  {h2h.source_a}: {h2h.a_wins} ({pct_a:.0f}%)  "
            f"{h2h.source_b}: {h2h.b_wins} ({pct_b:.0f}%)  Ties: {h2h.ties}"
        )
        if h2h.mean_improvement > 0:
            lines.append(f"  Mean improvement when {h2h.source_a} wins: {h2h.mean_improvement:.2f}")
    return "\n".join(lines)


def _format_evaluation(se: SourceEvaluation, include_strata: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"Source: {se.source_name} ({se.year})")
    lines.append(_format_stat_accuracy_table("Batting", se.batting_stat_accuracy))
    lines.append(_format_rank_accuracy("Batting", se.batting_rank_accuracy))
    if include_strata and se.batting_strata:
        lines.append(_format_strata(se.batting_strata, "Batting"))
    lines.append(_format_stat_accuracy_table("Pitching", se.pitching_stat_accuracy))
    lines.append(_format_rank_accuracy("Pitching", se.pitching_rank_accuracy))
    if include_strata and se.pitching_strata:
        lines.append(_format_strata(se.pitching_strata, "Pitching"))
    return "\n".join(lines)


def _build_source(engine: str, data_source: StatsDataSource, year: int) -> tuple[str, ProjectionSource]:
    if engine in PIPELINES:
        pipeline = PIPELINES[engine]()
        return (engine, PipelineProjectionSource(pipeline, data_source, year))
    raise ValueError(f"Unknown engine: {engine}")


def _average_evaluations(
    evaluations: list[SourceEvaluation],
) -> SourceEvaluation:
    """Average stat and rank accuracies across multiple years."""
    source_name = evaluations[0].source_name
    years_str = ",".join(str(e.year) for e in evaluations)

    # Average batting stat accuracy
    batting_accuracies: list[StatAccuracy] = []
    if evaluations[0].batting_stat_accuracy:
        for i, cat_acc in enumerate(evaluations[0].batting_stat_accuracy):
            all_rmse = [e.batting_stat_accuracy[i].rmse for e in evaluations if e.batting_stat_accuracy]
            all_mae = [e.batting_stat_accuracy[i].mae for e in evaluations if e.batting_stat_accuracy]
            all_corr = [e.batting_stat_accuracy[i].correlation for e in evaluations if e.batting_stat_accuracy]
            all_n = [e.batting_stat_accuracy[i].sample_size for e in evaluations if e.batting_stat_accuracy]
            batting_accuracies.append(
                StatAccuracy(
                    category=cat_acc.category,
                    sample_size=sum(all_n),
                    rmse=sum(all_rmse) / len(all_rmse),
                    mae=sum(all_mae) / len(all_mae),
                    correlation=sum(all_corr) / len(all_corr),
                )
            )

    # Average pitching stat accuracy
    pitching_accuracies: list[StatAccuracy] = []
    if evaluations[0].pitching_stat_accuracy:
        for i, cat_acc in enumerate(evaluations[0].pitching_stat_accuracy):
            all_rmse = [e.pitching_stat_accuracy[i].rmse for e in evaluations if e.pitching_stat_accuracy]
            all_mae = [e.pitching_stat_accuracy[i].mae for e in evaluations if e.pitching_stat_accuracy]
            all_corr = [e.pitching_stat_accuracy[i].correlation for e in evaluations if e.pitching_stat_accuracy]
            all_n = [e.pitching_stat_accuracy[i].sample_size for e in evaluations if e.pitching_stat_accuracy]
            pitching_accuracies.append(
                StatAccuracy(
                    category=cat_acc.category,
                    sample_size=sum(all_n),
                    rmse=sum(all_rmse) / len(all_rmse),
                    mae=sum(all_mae) / len(all_mae),
                    correlation=sum(all_corr) / len(all_corr),
                )
            )

    # Average rank accuracy
    batting_rank: RankAccuracy | None = None
    bat_ranks = [e.batting_rank_accuracy for e in evaluations if e.batting_rank_accuracy is not None]
    if bat_ranks:
        batting_rank = RankAccuracy(
            sample_size=sum(r.sample_size for r in bat_ranks),
            spearman_rho=sum(r.spearman_rho for r in bat_ranks) / len(bat_ranks),
            top_n=bat_ranks[0].top_n,
            top_n_precision=sum(r.top_n_precision for r in bat_ranks) / len(bat_ranks),
        )

    pitching_rank: RankAccuracy | None = None
    pitch_ranks = [e.pitching_rank_accuracy for e in evaluations if e.pitching_rank_accuracy is not None]
    if pitch_ranks:
        pitching_rank = RankAccuracy(
            sample_size=sum(r.sample_size for r in pitch_ranks),
            spearman_rho=sum(r.spearman_rho for r in pitch_ranks) / len(pitch_ranks),
            top_n=pitch_ranks[0].top_n,
            top_n_precision=sum(r.top_n_precision for r in pitch_ranks) / len(pitch_ranks),
        )

    return SourceEvaluation(
        source_name=source_name,
        year=int(years_str.split(",")[0]),
        batting_stat_accuracy=tuple(batting_accuracies),
        pitching_stat_accuracy=tuple(pitching_accuracies),
        batting_rank_accuracy=batting_rank,
        pitching_rank_accuracy=pitching_rank,
    )


def _evaluation_to_dict(se: SourceEvaluation) -> dict[str, object]:
    """Convert a SourceEvaluation to a JSON-serializable dict."""
    result: dict[str, object] = {
        "source_name": se.source_name,
        "year": se.year,
    }
    batting_stats: list[dict[str, object]] = []
    for sa in se.batting_stat_accuracy:
        batting_stats.append(
            {
                "category": sa.category.value,
                "sample_size": sa.sample_size,
                "rmse": round(sa.rmse, 4),
                "mae": round(sa.mae, 4),
                "correlation": round(sa.correlation, 4),
            }
        )
    result["batting_stat_accuracy"] = batting_stats

    pitching_stats: list[dict[str, object]] = []
    for sa in se.pitching_stat_accuracy:
        pitching_stats.append(
            {
                "category": sa.category.value,
                "sample_size": sa.sample_size,
                "rmse": round(sa.rmse, 4),
                "mae": round(sa.mae, 4),
                "correlation": round(sa.correlation, 4),
            }
        )
    result["pitching_stat_accuracy"] = pitching_stats

    if se.batting_rank_accuracy:
        result["batting_rank_accuracy"] = {
            "sample_size": se.batting_rank_accuracy.sample_size,
            "spearman_rho": round(se.batting_rank_accuracy.spearman_rho, 4),
            "top_n": se.batting_rank_accuracy.top_n,
            "top_n_precision": round(se.batting_rank_accuracy.top_n_precision, 4),
        }

    if se.pitching_rank_accuracy:
        result["pitching_rank_accuracy"] = {
            "sample_size": se.pitching_rank_accuracy.sample_size,
            "spearman_rho": round(se.pitching_rank_accuracy.spearman_rho, 4),
            "top_n": se.pitching_rank_accuracy.top_n,
            "top_n_precision": round(se.pitching_rank_accuracy.top_n_precision, 4),
        }

    return result


def evaluate_cmd(
    year: Annotated[int, typer.Argument(help="Season year to evaluate against.")],
    engine: Annotated[list[str] | None, typer.Option(help="Projection engine(s) to evaluate.")] = None,
    years: Annotated[
        str | None, typer.Option(help="Comma-separated years for multi-year backtest (e.g. 2021,2022,2023).")
    ] = None,
    min_pa: Annotated[int, typer.Option(help="Minimum plate appearances for inclusion.")] = 200,
    min_ip: Annotated[float, typer.Option(help="Minimum innings pitched for inclusion.")] = 50.0,
    top_n: Annotated[int, typer.Option(help="N for top-N precision metric.")] = 20,
    output_json: Annotated[str | None, typer.Option("--json", help="Write results to JSON file.")] = None,
    stratify: Annotated[bool, typer.Option(help="Enable stratification by PA/IP and age buckets.")] = False,
    include_residuals: Annotated[
        bool, typer.Option(help="Include per-player residuals (enables head-to-head).")
    ] = False,
    compare: Annotated[
        bool, typer.Option(help="Show head-to-head comparison (requires 2+ engines and --include-residuals).")
    ] = False,
) -> None:
    """Evaluate projection sources against actual season outcomes."""
    engines = engine or [DEFAULT_ENGINE]
    for eng in engines:
        validate_engine(eng)

    eval_years = [int(y.strip()) for y in years.split(",")] if years else [year]

    data_source = get_container().data_source
    league_settings = load_league_settings()

    # Build stratification config if requested
    stratification: StratificationConfig | None = None
    if stratify or include_residuals:
        stratification = StratificationConfig(include_residuals=include_residuals)

    all_results: list[dict[str, object]] = []
    all_evaluations: list[SourceEvaluation] = []

    for eng in engines:
        year_evaluations: list[SourceEvaluation] = []
        for eval_year in eval_years:
            config = EvaluationConfig(
                year=eval_year,
                batting_categories=league_settings.batting_categories,
                pitching_categories=league_settings.pitching_categories,
                min_pa=min_pa,
                min_ip=min_ip,
                top_n=top_n,
                stratification=stratification,
            )
            source = _build_source(eng, data_source, eval_year)
            result = evaluate(sources=[source], data_source=data_source, config=config)
            se = result.evaluations[0]
            year_evaluations.append(se)
            all_evaluations.append(se)

            if not years:
                typer.echo(_format_evaluation(se, include_strata=stratify))

        if years and len(year_evaluations) > 1:
            avg = _average_evaluations(year_evaluations)
            years_label = ", ".join(str(y) for y in eval_years)
            typer.echo(f"Average across {years_label}:")
            typer.echo(_format_evaluation(avg, include_strata=stratify))

        for se in year_evaluations:
            all_results.append(_evaluation_to_dict(se))

    # Head-to-head comparison if requested
    if compare:
        if len(engines) < 2:
            typer.echo("\nWarning: --compare requires at least 2 engines for head-to-head comparison.")
        elif not include_residuals:
            typer.echo("\nWarning: --compare requires --include-residuals for head-to-head comparison.")
        else:
            # Compare first two engines (using first year's evaluations for simplicity)
            engine_evals = {se.source_name: se for se in all_evaluations}
            if len(engine_evals) >= 2:
                eval_list = list(engine_evals.values())
                h2h_results = compare_sources(eval_list[0], eval_list[1])
                typer.echo("\n" + _format_head_to_head(h2h_results))

    if output_json:
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        typer.echo(f"\nResults written to {output_json}")
