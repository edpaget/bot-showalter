import csv
import datetime
import logging
import math
from pathlib import Path  # noqa: TC003 — used at runtime by typer
from typing import Annotated, Any

import typer
from rich.table import Table

from fantasy_baseball_manager.cli._output import console, print_error, print_ingest_result
from fantasy_baseball_manager.cli.factory import IngestContainer, build_ingest_container
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    Err,
    Ok,
    Player,
)
from fantasy_baseball_manager.ingest import (
    CsvSource,
    FantasyProsADPSource,
    Loader,
    chadwick_row_to_player,
    ingest_fantasypros_adp,
    ingest_roster_api,
    lahman_team_row_to_team,
    make_fg_batting_mapper,
    make_fg_pitching_mapper,
    make_il_stint_mapper,
    make_lahman_bio_mapper,
    make_milb_batting_mapper,
    make_position_appearance_mapper,
    make_roster_stint_mapper,
    make_sprint_speed_mapper,
    statcast_pitch_mapper,
)
from fantasy_baseball_manager.mlb_api import fetch_mlb_active_teams
from fantasy_baseball_manager.services import MlbApiPlayerTeamProvider

logger = logging.getLogger(__name__)

ingest_app = typer.Typer(name="ingest", help="Ingest historical player data and stats")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


@ingest_app.command("players")
def ingest_players(
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest player data from the Chadwick register."""
    with build_ingest_container(data_dir) as container:
        source = container.player_source()
        loader = Loader(
            source,
            container.player_repo,
            container.log_repo,
            chadwick_row_to_player,
            "player",
            provider=SingleConnectionProvider(container.conn),
        )
        match loader.load():
            case Ok(log):
                print_ingest_result(log)
            case Err(e):
                print_error(e.message)


@ingest_app.command("bio")
def ingest_bio(
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Enrich existing players with birth date, bats, and throws from Lahman."""
    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        source = container.bio_source()
        mapper = make_lahman_bio_mapper(players)
        loader = Loader(
            source,
            container.player_repo,
            container.log_repo,
            mapper,
            "player",
            provider=SingleConnectionProvider(container.conn),
        )
        match loader.load():
            case Ok(log):
                print_ingest_result(log)
            case Err(e):
                print_error(e.message)


@ingest_app.command("batting")
def ingest_batting(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest historical batting stats."""
    with build_ingest_container(data_dir) as container:
        data_source = container.batting_source()
        players = container.player_repo.all()
        for yr in season:
            mapper = make_fg_batting_mapper(players)
            loader = Loader(
                data_source,
                container.batting_stats_repo,
                container.log_repo,
                mapper,
                "batting_stats",
                provider=SingleConnectionProvider(container.conn),
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("pitching")
def ingest_pitching(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest historical pitching stats."""
    with build_ingest_container(data_dir) as container:
        data_source = container.pitching_source()
        players = container.player_repo.all()
        for yr in season:
            mapper = make_fg_pitching_mapper(players)
            loader = Loader(
                data_source,
                container.pitching_stats_repo,
                container.log_repo,
                mapper,
                "pitching_stats",
                provider=SingleConnectionProvider(container.conn),
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("statcast")
def ingest_statcast(  # pragma: no cover
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest Statcast pitch-level data."""
    with build_ingest_container(data_dir) as container:
        for yr in season:
            start_dt = f"{yr}-03-01"
            end_dt = f"{yr}-11-30"
            loader = Loader(
                container.statcast_source(),
                container.statcast_pitch_repo,
                container.log_repo,
                statcast_pitch_mapper,
                "statcast_pitch",
                provider=SingleConnectionProvider(container.statcast_conn),
                log_provider=SingleConnectionProvider(container.conn),
            )
            match loader.load(start_dt=start_dt, end_dt=end_dt):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("sprint-speed")
def ingest_sprint_speed(  # pragma: no cover
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest Baseball Savant sprint speed data."""
    with build_ingest_container(data_dir) as container:
        for yr in season:
            mapper = make_sprint_speed_mapper(season=yr)
            loader = Loader(
                container.sprint_speed_source(),
                container.sprint_speed_repo,
                container.log_repo,
                mapper,
                "sprint_speed",
                provider=SingleConnectionProvider(container.statcast_conn),
                log_provider=SingleConnectionProvider(container.conn),
            )
            match loader.load(year=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("il")
def ingest_il(  # pragma: no cover
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest IL transaction data from the MLB Stats API."""
    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        for yr in season:
            mapper = make_il_stint_mapper(players, season=yr)
            source = container.il_source()
            loader = Loader(
                source,
                container.il_stint_repo,
                container.log_repo,
                mapper,
                "il_stint",
                provider=SingleConnectionProvider(container.conn),
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("il-coverage")
def ingest_il_coverage(
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show IL stint data coverage by season."""
    with build_ingest_container(data_dir) as container:
        counts = container.il_stint_repo.count_by_season()
        if not counts:
            console.print("  No IL stint data found.")
            return
        table = Table(title="IL Stint Coverage")
        table.add_column("Season", justify="right")
        table.add_column("Stints", justify="right")
        table.add_column("Status")
        total = 0
        for season in sorted(counts):
            count = counts[season]
            total += count
            status = "[green]✓[/green]" if count > 0 else "[red]✗[/red]"
            table.add_row(str(season), str(count), status)
        table.add_section()
        table.add_row("Total", str(total), "")
        console.print(table)


@ingest_app.command("appearances")
def ingest_appearances(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest position appearance data from Lahman."""
    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        mapper = make_position_appearance_mapper(players)
        for yr in season:
            loader = Loader(
                container.appearances_source(),
                container.position_appearance_repo,
                container.log_repo,
                mapper,
                "position_appearance",
                provider=SingleConnectionProvider(container.conn),
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("roster")
def ingest_roster(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest roster stint data from Lahman."""
    with build_ingest_container(data_dir) as container:
        # Auto-upsert teams first
        teams_source = container.teams_source()
        for yr in season:
            teams_rows = teams_source.fetch(season=yr)
            for row in teams_rows:
                team = lahman_team_row_to_team(row)
                if team is not None:
                    container.team_repo.upsert(team)
            container.conn.commit()

        players = container.player_repo.all()
        teams = container.team_repo.all()
        mapper = make_roster_stint_mapper(players, teams)
        for yr in season:
            loader = Loader(
                container.appearances_source(),
                container.roster_stint_repo,
                container.log_repo,
                mapper,
                "roster_stint",
                provider=SingleConnectionProvider(container.conn),
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("roster-api")
def ingest_roster_api_cmd(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    as_of: Annotated[str | None, typer.Option("--as-of", help="Snapshot date (ISO)")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Pre-populate roster stints from the MLB Stats API."""
    if as_of is None:
        as_of = datetime.date.today().isoformat()

    with build_ingest_container(data_dir) as container:
        result = ingest_roster_api(
            fetch_mlb_active_teams,
            container.player_repo,
            container.team_repo,
            container.roster_stint_repo,
            season=season,
            as_of=as_of,
        )
        container.conn.commit()
        console.print(f"  Loaded {result.loaded} roster stints, skipped {result.skipped}")


def _auto_register_players(rows: list[dict[str, Any]], container: IngestContainer) -> None:
    """Register any players in the rows that aren't already in the player table."""
    if not rows:
        return
    existing_mlbam_ids = {p.mlbam_id for p in container.player_repo.all() if p.mlbam_id is not None}
    registered = 0
    for row in rows:
        mlbam_id = row.get("mlbam_id")
        if mlbam_id is None or (isinstance(mlbam_id, float) and math.isnan(mlbam_id)):
            continue
        if int(mlbam_id) in existing_mlbam_ids:
            continue
        first = row.get("first_name", "")
        last = row.get("last_name", "")
        container.player_repo.upsert(Player(name_first=str(first), name_last=str(last), mlbam_id=int(mlbam_id)))
        existing_mlbam_ids.add(int(mlbam_id))
        registered += 1
    if registered:
        container.conn.commit()


class _PreloadedSource:
    """Wraps pre-fetched rows so Loader can use them without re-fetching."""

    def __init__(self, rows: list[dict[str, Any]], original: object) -> None:
        self._rows = rows
        self._source_type = getattr(original, "source_type", "unknown")
        self._source_detail = getattr(original, "source_detail", "unknown")

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def source_detail(self) -> str:
        return self._source_detail

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        return self._rows


_MILB_LEVELS = ["AAA", "AA", "A+", "A", "ROK"]


@ingest_app.command("milb-batting")
def ingest_milb_batting(  # pragma: no cover
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    level: Annotated[list[str] | None, typer.Option("--level", help="Level(s): AAA, AA, A+, A, ROK")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest minor league batting stats from the MLB Stats API."""
    levels = level if level else _MILB_LEVELS
    with build_ingest_container(data_dir) as container:
        for yr in season:
            for lvl in levels:
                source = container.milb_batting_source()
                rows = source.fetch(season=yr, level=lvl)
                _auto_register_players(rows, container)
                players = container.player_repo.all()
                mapper = make_milb_batting_mapper(players)
                loader = Loader(
                    _PreloadedSource(rows, source),
                    container.minor_league_batting_stats_repo,
                    container.log_repo,
                    mapper,
                    "minor_league_batting_stats",
                    provider=SingleConnectionProvider(container.conn),
                )
                match loader.load(season=yr, level=lvl):
                    case Ok(log):
                        print_ingest_result(log)
                    case Err(e):
                        print_error(e.message)
                        continue


def _build_player_teams(container: IngestContainer, season: int) -> dict[int, str]:
    """Build a player_id -> team abbreviation mapping, preferring live MLB API data."""
    provider = MlbApiPlayerTeamProvider(
        player_repo=container.player_repo,
        team_repo=container.team_repo,
        roster_stint_repo=container.roster_stint_repo,
        fetcher=fetch_mlb_active_teams,
    )
    return provider.get_player_teams(season)


@ingest_app.command("adp")
def ingest_adp(  # pragma: no cover
    csv_path: Annotated[Path, typer.Argument(help="Path to FantasyPros ADP CSV")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    as_of: Annotated[str | None, typer.Option("--as-of", help="Snapshot date (ISO)")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest ADP data from a FantasyPros CSV file."""
    if not csv_path.exists():
        print_error(f"file not found: {csv_path}")
        raise typer.Exit(code=1)

    with build_ingest_container(data_dir) as container:
        source = CsvSource(csv_path)
        rows = source.fetch()
        players = container.player_repo.all()
        player_teams = _build_player_teams(container, season)
        result = ingest_fantasypros_adp(
            rows,
            container.adp_repo,
            players,
            season=season,
            as_of=as_of,
            player_teams=player_teams,
            player_repo=container.player_repo,
        )
        container.conn.commit()
        console.print(f"  Loaded {result.loaded} ADP records, skipped {result.skipped}")
        if result.created:
            console.print(f"  Created {result.created} player stubs for unknown players")
        if result.unmatched:
            console.print(
                f"  [yellow]Unmatched players ({len(result.unmatched)}):[/yellow] {', '.join(result.unmatched[:20])}"
            )
            if len(result.unmatched) > 20:
                console.print(f"  ... and {len(result.unmatched) - 20} more")


@ingest_app.command("adp-bulk")
def ingest_adp_bulk(  # pragma: no cover
    directory: Annotated[Path, typer.Argument(help="Directory with fantasypros_*.csv files")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Bulk-ingest ADP data from all fantasypros_*.csv files in a directory."""
    if not directory.is_dir():
        print_error(f"not a directory: {directory}")
        raise typer.Exit(code=1)

    csv_files = sorted(directory.glob("fantasypros_*.csv"))
    if not csv_files:
        print_error(f"no fantasypros_*.csv files found in {directory}")
        raise typer.Exit(code=1)

    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        for csv_file in csv_files:
            stem = csv_file.stem
            year_str = stem.replace("fantasypros_", "")
            try:
                season = int(year_str)
            except ValueError:
                console.print(f"  [yellow]Skipping {csv_file.name}: cannot parse season[/yellow]")
                continue

            source = CsvSource(csv_file)
            rows = source.fetch()
            player_teams = _build_player_teams(container, season)
            result = ingest_fantasypros_adp(
                rows,
                container.adp_repo,
                players,
                season=season,
                player_teams=player_teams,
                player_repo=container.player_repo,
            )
            container.conn.commit()
            n_unmatched = len(result.unmatched)
            parts = [f"loaded {result.loaded}", f"skipped {result.skipped}", f"unmatched {n_unmatched}"]
            if result.created:
                parts.append(f"created {result.created}")
            console.print(f"  {csv_file.name}: {', '.join(parts)}")


def _write_adp_csv(rows: list[dict[str, Any]], path: Path) -> None:  # pragma: no cover
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@ingest_app.command("adp-fetch")
def ingest_adp_fetch(  # pragma: no cover
    season: Annotated[int, typer.Option("--season", help="Season year")],
    as_of: Annotated[str | None, typer.Option("--as-of", help="Snapshot date (ISO)")] = None,
    save_csv: Annotated[Path | None, typer.Option("--save-csv", help="Save fetched data to CSV")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Fetch live ADP data from FantasyPros and ingest it."""
    if as_of is None:
        as_of = datetime.date.today().isoformat()

    source = FantasyProsADPSource()
    rows = source.fetch()
    console.print(f"  Fetched {len(rows)} rows from FantasyPros")

    if save_csv is not None:
        _write_adp_csv(rows, save_csv)
        console.print(f"  Saved CSV to {save_csv}")

    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        player_teams = _build_player_teams(container, season)
        result = ingest_fantasypros_adp(
            rows,
            container.adp_repo,
            players,
            season=season,
            as_of=as_of,
            player_teams=player_teams,
            player_repo=container.player_repo,
        )
        container.conn.commit()
        console.print(f"  Loaded {result.loaded} ADP records, skipped {result.skipped}")
        if result.created:
            console.print(f"  Created {result.created} player stubs for unknown players")
        if result.unmatched:
            console.print(
                f"  [yellow]Unmatched players ({len(result.unmatched)}):[/yellow] {', '.join(result.unmatched[:20])}"
            )
            if len(result.unmatched) > 20:
                console.print(f"  ... and {len(result.unmatched) - 20} more")
