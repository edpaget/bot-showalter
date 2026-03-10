from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from fantasy_baseball_manager.cli.factory import build_sgp_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.services import compute_sgp_denominators, find_league_lineage

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CategoryConfig, SgpDenominators

sgp_app = typer.Typer(name="sgp", help="SGP valuation tools")


@sgp_app.command("denominators")
def sgp_denominators(
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "h2h",
    seasons: Annotated[int | None, typer.Option("--seasons", help="Limit to last N seasons")] = None,
    yahoo_league: Annotated[str | None, typer.Option("--yahoo-league", help="Starting Yahoo league key")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Compute and display SGP denominators from league standings."""
    settings = load_league(league_name, Path.cwd())
    all_categories = list(settings.batting_categories) + list(settings.pitching_categories)

    with build_sgp_context(data_dir) as ctx:
        all_leagues = ctx.yahoo_league_repo.get_all()

        # Find start key: use provided yahoo-league or most recent non-keeper league
        if yahoo_league:
            start_key = yahoo_league
        else:
            redraft_leagues = [lg for lg in all_leagues if not lg.is_keeper]
            if not redraft_leagues:
                typer.echo("No redraft leagues found in database.", err=True)
                raise typer.Exit(1)
            redraft_leagues.sort(key=lambda lg: lg.season, reverse=True)
            start_key = redraft_leagues[0].league_key

        lineage_keys = find_league_lineage(all_leagues, start_key)
        if not lineage_keys:
            typer.echo(f"No league lineage found for {start_key}", err=True)
            raise typer.Exit(1)

        # Gather standings for all seasons in the lineage
        all_standings = []
        for league_key in lineage_keys:
            league = next((lg for lg in all_leagues if lg.league_key == league_key), None)
            if league is None:
                continue
            team_stats = ctx.yahoo_team_stats_repo.get_by_league_season(league_key, league.season)
            all_standings.extend(team_stats)

        if seasons is not None:
            all_seasons_list = sorted({ts.season for ts in all_standings})
            keep_seasons = set(all_seasons_list[-seasons:])
            all_standings = [ts for ts in all_standings if ts.season in keep_seasons]

        if not all_standings:
            typer.echo("No standings data found.", err=True)
            raise typer.Exit(1)

        result = compute_sgp_denominators(all_standings, all_categories)

    _print_denominators(result, all_categories)


def _print_denominators(result: SgpDenominators, categories: list[CategoryConfig]) -> None:
    season_set = sorted({sd.season for sd in result.per_season})

    header = f"{'Category':<10} {'Avg':>8}"
    for s in season_set:
        header += f" {s:>8}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for cat in categories:
        if cat.key not in result.averages:
            continue
        avg = result.averages[cat.key]
        row = f"{cat.key:<10} {avg:>8.3f}"
        for s in season_set:
            season_val = next(
                (sd.denominator for sd in result.per_season if sd.category == cat.key and sd.season == s),
                None,
            )
            if season_val is not None:
                row += f" {season_val:>8.3f}"
            else:
                row += f" {'—':>8}"
        typer.echo(row)
