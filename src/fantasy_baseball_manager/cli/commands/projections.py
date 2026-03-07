import datetime
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    print_error,
    print_import_result,
    print_player_projections,
    print_pt_sources,
    print_system_summaries,
)
from fantasy_baseball_manager.cli.factory import build_import_context, build_projections_context
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import Err, Ok, Projection
from fantasy_baseball_manager.ingest import (
    PROJECTION_SYSTEMS,
    FgProjectionSource,
    Loader,
    make_fg_projection_batting_mapper,
    make_fg_projection_pitching_mapper,
)

projections_app = typer.Typer(name="projections", help="Look up and explore projection systems")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


@projections_app.command("lookup")
def projections_lookup(
    player_name: Annotated[str, typer.Argument(help="Player name ('Last' or 'Last, First')")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by system")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Look up a player's projections across systems."""
    with build_projections_context(data_dir) as ctx:
        results = ctx.lookup_service.lookup(player_name, season, system=system)
    print_player_projections(results)


@projections_app.command("systems")
def projections_systems(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List available projection systems for a season."""
    with build_projections_context(data_dir) as ctx:
        summaries = ctx.lookup_service.list_systems(season)
    print_system_summaries(summaries)


@projections_app.command("pt-sources")
def projections_pt_sources(  # pragma: no cover
    season: Annotated[int, typer.Option("--season", help="Season year")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List projection systems that provide playing-time (PA/IP) data."""
    with build_projections_context(data_dir) as ctx:
        sources = ctx.lookup_service.list_pt_sources(season)
    print_pt_sources(sources)


@projections_app.command("sync")
def projections_sync(
    system: Annotated[str | None, typer.Argument(help="System: fangraphs-dc, steamer, zips")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = ...,  # type: ignore[assignment]
    version: Annotated[str | None, typer.Option("--version", help="Version (default: today's date)")] = None,
    all_systems: Annotated[bool, typer.Option("--all", help="Sync all systems")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Sync FanGraphs projections from the API into the database."""
    if not system and not all_systems:
        print_error("Provide a system name or use --all")
        raise typer.Exit(code=1)
    if system and all_systems:
        print_error("Provide either a system name or --all, not both")
        raise typer.Exit(code=1)

    if all_systems:
        systems_to_sync = list(PROJECTION_SYSTEMS.keys())
    else:
        assert system is not None  # noqa: S101
        if system not in PROJECTION_SYSTEMS:
            print_error(f"Unknown system {system!r}; expected one of: {', '.join(sorted(PROJECTION_SYSTEMS))}")
            raise typer.Exit(code=1)
        systems_to_sync = [system]

    resolved_version = version or datetime.date.today().isoformat()

    with build_import_context(data_dir) as ctx:
        players = ctx.player_repo.all()

        for sys_name in systems_to_sync:
            api_type = PROJECTION_SYSTEMS[sys_name]

            for stat_type, player_type, make_mapper in [
                ("bat", "batter", make_fg_projection_batting_mapper),
                ("pit", "pitcher", make_fg_projection_pitching_mapper),
            ]:
                source = FgProjectionSource(projection_type=api_type, stat_type=stat_type)
                mapper = make_mapper(
                    players,
                    season=season,
                    system=sys_name,
                    version=resolved_version,
                    source_type="third_party",
                )

                def _post_upsert(projection_id: int, projection: Projection) -> None:
                    if projection.distributions is not None:
                        ctx.proj_repo.upsert_distributions(projection_id, list(projection.distributions.values()))

                loader = Loader(
                    source,
                    ctx.proj_repo,
                    ctx.log_repo,
                    mapper,
                    "projection",
                    provider=SingleConnectionProvider(ctx.conn),
                    post_upsert=_post_upsert,
                )
                match loader.load():
                    case Ok(log):
                        print_import_result(log)
                    case Err(e):
                        print_error(f"{sys_name} {player_type}: {e.message}")
