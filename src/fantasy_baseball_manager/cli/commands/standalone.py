import os
from pathlib import Path  # noqa: TC003 — used at runtime by typer
from typing import Annotated

import typer

from fantasy_baseball_manager.agent.chat import run_chat
from fantasy_baseball_manager.agent.graph import build_agent
from fantasy_baseball_manager.agent.prompt import current_season
from fantasy_baseball_manager.cli._output import (
    console,
    print_comparison_result,
    print_error,
    print_features,
    print_import_result,
    print_regression_check_result,
    print_stratified_comparison_result,
)
from fantasy_baseball_manager.cli.factory import (
    EvalContext,
    build_chat_context,
    build_eval_context,
    build_import_context,
    create_model,
)
from fantasy_baseball_manager.discord_bot.bot import FBMDiscordBot
from fantasy_baseball_manager.domain import (
    BATTING_RATE_STATS,
    PITCHING_RATE_STATS,
    Err,
    Ok,
    Projection,
    build_consensus_lookup,
    check_regression,
    summarize_comparison,
)
from fantasy_baseball_manager.ingest import (
    CsvSource,
    Loader,
    make_fg_projection_batting_mapper,
    make_fg_projection_pitching_mapper,
)
from fantasy_baseball_manager.models import FeatureIntrospectable, list_models
from fantasy_baseball_manager.services import (
    assign_age_cohorts,
    assign_experience_cohorts,
    assign_top300_cohorts,
)

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]
_TopOpt = Annotated[int | None, typer.Option("--top", help="Top N players by WAR to include")]
_ModelArg = Annotated[str, typer.Argument(help="Name of the projection model")]

_STRATIFY_CHOICES = ["age", "experience", "top300"]


def _build_cohort_assignments(ctx: EvalContext, dimension: str, season: int) -> dict[int, str]:
    """Build cohort assignment dict for the given stratification dimension."""
    if dimension == "age":
        players = ctx.player_repo.all()
        players_by_id = {p.id: p for p in players if p.id is not None}
        return assign_age_cohorts(players_by_id, season)
    if dimension == "experience":
        batting_actuals = ctx.batting_repo.get_by_season(season, source="fangraphs")
        player_ids = {a.player_id for a in batting_actuals}
        prior_batting = []
        for yr in range(season - 20, season):
            prior_batting.extend(ctx.batting_repo.get_by_season(yr, source="fangraphs"))
        return assign_experience_cohorts(prior_batting, player_ids)
    if dimension == "top300":
        batting_actuals = ctx.batting_repo.get_by_season(season, source="fangraphs")
        return assign_top300_cohorts(batting_actuals)
    msg = f"unknown dimension: {dimension}"
    raise ValueError(msg)


def list_cmd() -> None:
    """List all registered projection models."""
    names = list_models()
    if not names:
        console.print("No models registered.")
        return
    console.print("[bold]Registered models:[/bold]")
    for name in names:
        console.print(f"  {name}")


def info(model: _ModelArg) -> None:
    """Show metadata and supported operations for a model."""
    match create_model(model):
        case Ok(m):
            console.print(f"[bold]Model:[/bold] {m.name}")
            console.print(f"[bold]Description:[/bold] {m.description}")
            console.print(f"[bold]Operations:[/bold] {', '.join(sorted(m.supported_operations))}")
        case Err(e):
            print_error(e.message)
            raise typer.Exit(code=1)


def features(model: _ModelArg) -> None:
    """List declared features for a model."""
    match create_model(model):
        case Ok(m):
            if not isinstance(m, FeatureIntrospectable):
                print_error(f"model '{model}' does not expose features")
                raise typer.Exit(code=1)
            print_features(m.name, m.declared_features)
        case Err(e):
            print_error(e.message)
            raise typer.Exit(code=1)


def import_cmd(
    system: Annotated[str, typer.Argument(help="Projection system name (e.g. steamer, zips, atc)")],
    csv_path: Annotated[Path, typer.Argument(help="Path to CSV file")],
    version: Annotated[str, typer.Option("--version", help="Projection version")],
    player_type: Annotated[str, typer.Option("--player-type", help="Player type: batter or pitcher")],
    season: Annotated[int, typer.Option("--season", help="Season year for the projections")],
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Import third-party projections from a CSV file."""
    if not csv_path.exists():
        print_error(f"file not found: {csv_path}")
        raise typer.Exit(code=1)

    with build_import_context(data_dir) as ctx:
        players = ctx.player_repo.all()

        if player_type == "pitcher":
            mapper = make_fg_projection_pitching_mapper(
                players,
                season=season,
                system=system,
                version=version,
                source_type="third_party",
            )
        else:
            mapper = make_fg_projection_batting_mapper(
                players,
                season=season,
                system=system,
                version=version,
                source_type="third_party",
            )

        source = CsvSource(csv_path)

        def _post_upsert(projection_id: int, projection: Projection) -> None:
            if projection.distributions is not None:
                ctx.proj_repo.upsert_distributions(projection_id, list(projection.distributions.values()))

        loader = Loader(
            source, ctx.proj_repo, ctx.log_repo, mapper, "projection", conn=ctx.conn, post_upsert=_post_upsert
        )
        match loader.load(encoding="utf-8-sig"):
            case Ok(log):
                print_import_result(log)
            case Err(e):
                print_error(e.message)


def compare_cmd(
    systems: Annotated[list[str], typer.Argument(help="Systems to compare (format: system/version)")],
    season: Annotated[int, typer.Option("--season", help="Season to compare against")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to compare")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
    top: _TopOpt = None,
    stratify: Annotated[str | None, typer.Option("--stratify", help="Stratify by: age, experience, top300")] = None,
    normalize_pt: Annotated[str | None, typer.Option("--normalize-pt", help="PT source: consensus")] = None,
    rate_only: Annotated[bool, typer.Option("--rate-only", help="Evaluate rate stats only")] = False,
    min_pa: Annotated[int | None, typer.Option("--min-pa", help="Minimum PA for batters")] = None,
    min_ip: Annotated[int | None, typer.Option("--min-ip", help="Minimum IP for pitchers")] = None,
    tail: Annotated[bool, typer.Option("--tail", help="Show top-N tail accuracy")] = False,
    check: Annotated[bool, typer.Option("--check", help="Exit non-zero on regression")] = False,
) -> None:
    """Compare multiple projection systems against actuals."""
    if check:
        if len(systems) != 2:
            print_error("--check requires exactly 2 systems")
            raise typer.Exit(code=1)
        if stratify is not None:
            print_error("--check is incompatible with --stratify")
            raise typer.Exit(code=1)

    if stratify is not None and stratify not in _STRATIFY_CHOICES:
        print_error(f"invalid stratify dimension '{stratify}', expected one of: {', '.join(_STRATIFY_CHOICES)}")
        raise typer.Exit(code=1)

    if normalize_pt is not None and normalize_pt != "consensus":
        print_error(f"invalid --normalize-pt value '{normalize_pt}', expected 'consensus'")
        raise typer.Exit(code=1)

    parsed: list[tuple[str, str]] = []
    for s in systems:
        parts = s.split("/", 1)
        if len(parts) != 2:
            print_error(f"invalid system format '{s}', expected 'system/version'")
            raise typer.Exit(code=1)
        parsed.append((parts[0], parts[1]))

    if rate_only and stat is None:
        stat = list(BATTING_RATE_STATS + PITCHING_RATE_STATS)

    with build_eval_context(data_dir) as ctx:
        consensus = None
        if normalize_pt == "consensus":
            steamer_projs = ctx.projection_repo.get_by_season(season, system="steamer")
            zips_projs = ctx.projection_repo.get_by_season(season, system="zips")
            consensus = build_consensus_lookup(steamer_projs, zips_projs)

        tail_ns = (25, 50) if tail else None
        if stratify is None:
            result = ctx.evaluator.compare(
                parsed,
                season,
                stats=stat,
                top=top,
                normalize_pt=consensus,
                min_pa=min_pa,
                min_ip=min_ip,
                tail_ns=tail_ns,
            )
            print_comparison_result(result)
            if check:
                summary = summarize_comparison(result)
                check_result = check_regression(summary)
                print_regression_check_result(check_result)
                if not check_result.passed:
                    raise typer.Exit(code=1)
        else:
            cohort_assignments = _build_cohort_assignments(ctx, stratify, season)
            strat_result = ctx.evaluator.compare_stratified(
                parsed,
                season,
                cohort_assignments,
                dimension=stratify,
                stats=stat,
                top=top,
                normalize_pt=consensus,
                min_pa=min_pa,
                min_ip=min_ip,
            )
            print_stratified_comparison_result(strat_result)


def chat_cmd(  # pragma: no cover
    data_dir: _DataDirOpt = "./data",
    model: Annotated[str, typer.Option("--model", help="Anthropic model")] = "claude-haiku-4-5-20251001",
) -> None:
    """Start an interactive chat session with the fantasy baseball assistant."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("ANTHROPIC_API_KEY environment variable is required.")
        raise typer.Exit(code=1)
    with build_chat_context(data_dir) as ctx:
        agent = build_agent(ctx.container, season=current_season(), model=model)
        run_chat(agent)


def discord_cmd(  # pragma: no cover
    data_dir: _DataDirOpt = "./data",
    model: Annotated[str, typer.Option("--model", help="Anthropic model")] = "claude-haiku-4-5-20251001",
) -> None:
    """Run the Discord bot for the fantasy baseball assistant."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print_error("ANTHROPIC_API_KEY environment variable is required.")
        raise typer.Exit(code=1)
    token = os.environ.get("FBM_DISCORD_TOKEN")
    if not token:
        print_error("FBM_DISCORD_TOKEN environment variable is required.")
        raise typer.Exit(code=1)
    with build_chat_context(data_dir, check_same_thread=False) as ctx:
        agent = build_agent(ctx.container, season=current_season(), model=model)
        FBMDiscordBot(agent).run(token)
