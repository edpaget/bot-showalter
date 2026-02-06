from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — needed at runtime by Typer
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.statcast.models import DEFAULT_DATA_DIR, DownloadConfig

logger = logging.getLogger(__name__)
console = Console()

statcast_app = typer.Typer(help="Statcast pitch-level data commands.")


def _get_data_dir() -> Path:
    return DEFAULT_DATA_DIR


@statcast_app.command(name="download")
def download_cmd(
    seasons: Annotated[
        str,
        typer.Option(
            "--seasons",
            "-s",
            help="Comma-separated seasons to download (e.g., 2023,2024)",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Re-download all dates even if already fetched",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Override the default data directory",
        ),
    ] = None,
) -> None:
    """Download pitch-level Statcast data for the specified seasons.

    Example:
        uv run python -m fantasy_baseball_manager statcast download --seasons 2023,2024
        uv run python -m fantasy_baseball_manager statcast download --seasons 2024 --force
    """
    from fantasy_baseball_manager.statcast.downloader import StatcastDownloader
    from fantasy_baseball_manager.statcast.fetcher import PybaseballFetcher
    from fantasy_baseball_manager.statcast.models import ChunkResult
    from fantasy_baseball_manager.statcast.store import StatcastStore

    parsed_seasons = tuple(int(s.strip()) for s in seasons.split(","))
    resolved_dir = data_dir if data_dir is not None else _get_data_dir()

    config = DownloadConfig(
        seasons=parsed_seasons,
        data_dir=resolved_dir,
        force=force,
    )
    store = StatcastStore(data_dir=config.data_dir)
    fetcher = PybaseballFetcher()

    def on_progress(result: ChunkResult) -> None:
        status = "[green]OK[/green]" if result.success else "[red]FAIL[/red]"
        console.print(f"  {result.date.isoformat()} {status} ({result.row_count} rows)")

    downloader = StatcastDownloader(
        fetcher=fetcher,
        store=store,
        config=config,
        progress_callback=on_progress,
    )

    results = downloader.download_all()

    # Print summary table
    table = Table(title="Download Summary")
    table.add_column("Season")
    table.add_column("Status")
    table.add_column("Dates Fetched", justify="right")
    table.add_column("Total Rows", justify="right")

    for season in parsed_seasons:
        result = results[season]
        if result.is_ok():
            manifest = result.unwrap()
            table.add_row(
                str(season),
                "[green]Complete[/green]",
                str(len(manifest.fetched_dates)),
                str(manifest.total_rows),
            )
        else:
            table.add_row(str(season), "[red]Failed[/red]", "-", "-")

    console.print(table)


@statcast_app.command(name="status")
def status_cmd(
    season: Annotated[
        int | None,
        typer.Option(
            "--season",
            "-s",
            help="Show status for a specific season only",
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Override the default data directory",
        ),
    ] = None,
) -> None:
    """Show the status of downloaded Statcast data.

    Example:
        uv run python -m fantasy_baseball_manager statcast status
        uv run python -m fantasy_baseball_manager statcast status --season 2024
    """
    from fantasy_baseball_manager.statcast.store import StatcastStore

    resolved_dir = data_dir if data_dir is not None else _get_data_dir()
    store = StatcastStore(data_dir=resolved_dir)

    if season is not None:
        seasons = [season]
    else:
        # Discover seasons from directory listing
        if not resolved_dir.exists():
            console.print("No statcast data found.")
            return
        seasons = sorted(int(d.name) for d in resolved_dir.iterdir() if d.is_dir() and d.name.isdigit())

    if not seasons:
        console.print("No statcast data found.")
        return

    table = Table(title="Statcast Data Status")
    table.add_column("Season")
    table.add_column("Dates Fetched", justify="right")
    table.add_column("Total Rows", justify="right")
    table.add_column("Parquet Files", justify="right")
    table.add_column("Disk Size", justify="right")

    for s in seasons:
        manifest = store.load_manifest(s)
        season_dir = resolved_dir / str(s)
        parquet_files = list(season_dir.glob("statcast_*.parquet")) if season_dir.exists() else []
        disk_size = sum(f.stat().st_size for f in parquet_files)
        size_str = _format_size(disk_size)

        table.add_row(
            str(s),
            str(len(manifest.fetched_dates)),
            str(manifest.total_rows),
            str(len(parquet_files)),
            size_str,
        )

    console.print(table)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
