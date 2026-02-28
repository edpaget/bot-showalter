from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._helpers import parse_system_version
from fantasy_baseball_manager.cli._output import (
    console,
    print_adp_accuracy_report,
    print_adp_movers_report,
    print_error,
    print_performance_report,
    print_projection_confidence,
    print_residual_analysis_report,
    print_residual_persistence_report,
    print_system_disagreements,
    print_talent_delta_report,
    print_talent_quality_report,
    print_value_over_adp,
    print_variance_targets,
)
from fantasy_baseball_manager.cli.factory import (
    build_adp_accuracy_context,
    build_adp_movers_context,
    build_adp_report_context,
    build_confidence_report_context,
    build_report_context,
)
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import ConfidenceReport, VarianceClassification
from fantasy_baseball_manager.services import classify_variance, compute_confidence

report_app = typer.Typer(name="report", help="Over/underperformance reports vs model predictions")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


@report_app.command("overperformers")
def report_overperformers(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to report")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N rows")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show players who outperformed their expected stats."""
    sys_name, version = parse_system_version(system)

    with build_report_context(data_dir) as ctx:
        deltas = ctx.report_service.compute_deltas(
            sys_name,
            version,
            season,
            player_type,
            stats=stat,
        )

    overperformers = [d for d in deltas if d.performance_delta > 0]
    overperformers.sort(key=lambda d: d.performance_delta, reverse=True)
    if top is not None:
        overperformers = overperformers[:top]
    print_performance_report("Overperformers", overperformers)


@report_app.command("underperformers")
def report_underperformers(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to report")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N rows")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show players who underperformed their expected stats."""
    sys_name, version = parse_system_version(system)

    with build_report_context(data_dir) as ctx:
        deltas = ctx.report_service.compute_deltas(
            sys_name,
            version,
            season,
            player_type,
            stats=stat,
        )

    underperformers = [d for d in deltas if d.performance_delta < 0]
    underperformers.sort(key=lambda d: d.performance_delta)
    if top is not None:
        underperformers = underperformers[:top]
    print_performance_report("Underperformers", underperformers)


@report_app.command("talent-delta")
def report_talent_delta(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to include")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N per direction per stat")] = None,
    min_pa: Annotated[int | None, typer.Option("--min-pa", help="Minimum PA (batters) or IP (pitchers)")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show talent-delta report: regression candidates and buy-low targets."""
    sys_name, version = parse_system_version(system)

    with build_report_context(data_dir) as ctx:
        deltas = ctx.report_service.compute_deltas(
            sys_name,
            version,
            season,
            player_type,
            stats=stat,
            min_pa=min_pa,
        )

    pa_label = "IP" if player_type == "pitcher" else "PA"
    min_pa_str = f", min {min_pa} {pa_label}" if min_pa else ""
    title = f"Talent Delta — {system} vs {season} actuals ({player_type}s{min_pa_str})"
    print_talent_delta_report(title, deltas, top=top)


@report_app.command("talent-quality")
def report_talent_quality(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[list[int], typer.Option("--season", help="Two seasons (N and N+1)")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to evaluate")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate true-talent estimation quality across consecutive seasons."""
    if len(season) != 2:
        print_error("exactly two --season values required (N and N+1)")
        raise typer.Exit(code=1)

    sys_name, version = parse_system_version(system)

    season_n, season_n1 = sorted(season)

    with build_report_context(data_dir) as ctx:
        reports = []
        for player_type in ("batter", "pitcher"):
            report = ctx.talent_evaluator.evaluate(sys_name, version, season_n, season_n1, player_type, stats=stat)
            if report.stat_metrics:
                reports.append(report)

    if not reports:
        console.print("No projections found for either player type.")
        raise typer.Exit(code=1)

    print_talent_quality_report(reports)


@report_app.command("residual-persistence")
def report_residual_persistence(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[list[int], typer.Option("--season", help="Two seasons (N and N+1)")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to evaluate")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Diagnose residual persistence for batter projections across consecutive seasons."""
    if len(season) != 2:
        print_error("exactly two --season values required (N and N+1)")
        raise typer.Exit(code=1)

    sys_name, version = parse_system_version(system)

    season_n, season_n1 = sorted(season)

    with build_report_context(data_dir) as ctx:
        report = ctx.residual_diagnostic.diagnose(sys_name, version, season_n, season_n1, stats=stat)

    if not report.stat_metrics:
        console.print("No batter projections found.")
        raise typer.Exit(code=1)

    print_residual_persistence_report(report)


@report_app.command("residual-analysis")
def report_residual_analysis(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm-preseason/phase4)")],
    season: Annotated[list[int], typer.Option("--season", help="Season(s) to analyze")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to evaluate")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Filter to top-N by actual WAR")] = None,
    min_pa: Annotated[int | None, typer.Option("--min-pa", help="Minimum PA for batters")] = None,
    min_ip: Annotated[int | None, typer.Option("--min-ip", help="Minimum IP for pitchers")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Analyze systematic prediction bias and heteroscedasticity."""
    sys_name, version = parse_system_version(system)

    with build_report_context(data_dir) as ctx:
        report = ctx.residual_analysis_diagnostic.analyze(
            sys_name,
            version,
            seasons=season,
            stats=stat,
            top=top,
            min_pa=min_pa,
            min_ip=min_ip,
        )

    if not report.stat_analyses:
        console.print("No projections found.")
        raise typer.Exit(code=1)

    print_residual_analysis_report(report)


@report_app.command("value-over-adp")
def report_value_over_adp(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    player_type: Annotated[str | None, typer.Option("--player-type")] = None,
    position: Annotated[str | None, typer.Option("--position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N per section")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show value-over-ADP report: buy targets, avoids, and sleepers."""
    with build_adp_report_context(data_dir) as ctx:
        report = ctx.service.compute_value_over_adp(
            season,
            system,
            version,
            provider=provider,
            player_type=player_type,
            position=position,
            top=top,
        )
    print_value_over_adp(report)


@report_app.command("adp-accuracy")
def report_adp_accuracy(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    compare_system: Annotated[str | None, typer.Option("--compare-system", help="Valuation system/version")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate how well ADP predicts actual season outcomes."""
    league = load_league(league_name, Path.cwd())

    compare: tuple[str, str] | None = None
    if compare_system is not None:
        parts = compare_system.split("/", 1)
        if len(parts) != 2:
            print_error(f"invalid compare-system format '{compare_system}', expected 'system/version'")
            raise typer.Exit(code=1)
        compare = (parts[0], parts[1])

    with build_adp_accuracy_context(data_dir) as ctx:
        report = ctx.evaluator.evaluate(season, league, provider=provider, compare_system=compare)
    print_adp_accuracy_report(report)


@report_app.command("adp-movers")
def report_adp_movers(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    window: Annotated[int, typer.Option("--window", help="Days before latest snapshot")] = 14,
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    top: Annotated[int, typer.Option("--top", help="Show top N risers/fallers")] = 20,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show the biggest ADP movers between two snapshots."""
    with build_adp_movers_context(data_dir) as ctx:
        try:
            current, previous = ctx.service.resolve_window(season, provider, window)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None
        report = ctx.service.compute_adp_movers(season, provider, current, previous, top=top)
    print_adp_movers_report(report)


@report_app.command("projection-confidence")
def report_projection_confidence(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    min_systems: Annotated[int, typer.Option("--min-systems", help="Minimum projection systems required")] = 3,
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter: batter or pitcher")] = None,
    agreement: Annotated[str | None, typer.Option("--agreement", help="Filter: high, medium, or low")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show projection confidence report: cross-system agreement per player."""
    league = load_league(league_name, Path.cwd())

    with build_confidence_report_context(data_dir) as ctx:
        projections = ctx.projection_repo.get_by_season(season)
        if player_type is not None:
            projections = [p for p in projections if p.player_type == player_type]

        player_ids = {p.player_id for p in projections}
        players = ctx.player_repo.get_by_ids(list(player_ids))
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        report = compute_confidence(
            projections,
            league,
            player_names,
            min_systems=min_systems,
        )

    if agreement is not None:
        report = ConfidenceReport(
            season=report.season,
            systems=report.systems,
            players=[p for p in report.players if p.agreement_level == agreement],
        )

    if top is not None:
        report = ConfidenceReport(
            season=report.season,
            systems=report.systems,
            players=report.players[:top],
        )

    print_projection_confidence(report)


@report_app.command("variance-targets")
def report_variance_targets(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    min_systems: Annotated[int, typer.Option("--min-systems")] = 3,
    player_type: Annotated[str | None, typer.Option("--player-type")] = None,
    classification: Annotated[str | None, typer.Option("--classification", help="Filter by classification")] = None,
    top: Annotated[int | None, typer.Option("--top")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show variance targets: players grouped into draft-actionable classification buckets."""
    league = load_league(league_name, Path.cwd())

    with build_confidence_report_context(data_dir) as ctx:
        projections = ctx.projection_repo.get_by_season(season)
        if player_type is not None:
            projections = [p for p in projections if p.player_type == player_type]

        player_ids = {p.player_id for p in projections}
        players = ctx.player_repo.get_by_ids(list(player_ids))
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        report = compute_confidence(
            projections,
            league,
            player_names,
            min_systems=min_systems,
        )

        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)

        classified = classify_variance(report, valuations, adp_list if adp_list else None)

    if classification is not None:
        try:
            target_cls = VarianceClassification(classification)
        except ValueError:
            print_error(
                f"invalid classification '{classification}', expected one of: "
                + ", ".join(c.value for c in VarianceClassification)
            )
            raise typer.Exit(code=1) from None
        classified = [c for c in classified if c.classification == target_cls]

    if top is not None:
        classified = classified[:top]

    print_variance_targets(classified)


@report_app.command("system-disagreements")
def report_system_disagreements(
    player: Annotated[str, typer.Option("--player", help="Player name (partial match)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    min_systems: Annotated[int, typer.Option("--min-systems")] = 3,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show per-system stat comparison for a single player."""
    league = load_league(league_name, Path.cwd())

    with build_confidence_report_context(data_dir) as ctx:
        all_projections = ctx.projection_repo.get_by_season(season)

        player_ids = {p.player_id for p in all_projections}
        players_list = ctx.player_repo.get_by_ids(list(player_ids))
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players_list if p.id is not None}

        # Search for player by name (case-insensitive partial match)
        search = player.lower()
        matches = [(pid, name) for pid, name in player_names.items() if search in name.lower()]

        if not matches:
            print_error(f"no player found matching '{player}'")
            raise typer.Exit(code=1)

        if len(matches) > 1:
            # Try exact match first
            exact = [(pid, name) for pid, name in matches if name.lower() == search]
            if len(exact) == 1:
                matches = exact
            else:
                console.print(f"Multiple players match '{player}':")
                for _, name in matches[:10]:
                    console.print(f"  {name}")
                raise typer.Exit(code=1)

        matched_pid, _ = matches[0]

        # Filter projections to matched player
        player_projections = [p for p in all_projections if p.player_id == matched_pid]

        report = compute_confidence(
            player_projections,
            league,
            player_names,
            min_systems=min_systems,
        )

    if not report.players:
        print_error(f"not enough projection systems for this player (need {min_systems})")
        raise typer.Exit(code=1)

    player_confidence = report.players[0]
    print_system_disagreements(player_confidence, player_projections)
