from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, DEFAULT_METHOD, validate_engine, validate_method
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.services import get_container, set_container
from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching

console = Console()

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection

__all__ = ["set_container", "valuate"]

_CATEGORY_MAP: dict[str, StatCategory] = {member.value.lower(): member for member in StatCategory}

_SUPPORTED_BATTING: set[StatCategory] = {
    StatCategory.HR,
    StatCategory.SB,
    StatCategory.OBP,
    StatCategory.R,
    StatCategory.RBI,
}
_SUPPORTED_PITCHING: set[StatCategory] = {StatCategory.K, StatCategory.ERA, StatCategory.WHIP}


def _parse_categories(
    raw: str,
    supported: set[StatCategory],
) -> tuple[StatCategory, ...]:
    categories: list[StatCategory] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token not in _CATEGORY_MAP:
            typer.echo(f"Unknown category: {token}", err=True)
            raise typer.Exit(code=1)
        cat = _CATEGORY_MAP[token]
        if cat not in supported:
            typer.echo(
                f"Category {cat.value} is not yet supported for valuation.",
                err=True,
            )
            raise typer.Exit(code=1)
        categories.append(cat)
    return tuple(categories)


def _print_value_table(
    title: str,
    categories: tuple[StatCategory, ...],
    values: list[PlayerValue],
    top: int,
) -> None:
    table = Table(title=title)
    table.add_column("Name")
    for cat in categories:
        table.add_column(cat.value, justify="right")
    table.add_column("Total", justify="right")

    for pv in values[:top]:
        row = [pv.name]
        for cv in pv.category_values:
            row.append(f"{cv.value:.1f}")
        row.append(f"{pv.total_value:.1f}")
        table.add_row(*row)

    console.print(table)


def valuate(
    year: Annotated[int | None, typer.Argument(help="Valuation year (default: current year).")] = None,
    batting: Annotated[bool, typer.Option("--batting", help="Show only batting values.")] = False,
    pitching: Annotated[bool, typer.Option("--pitching", help="Show only pitching values.")] = False,
    top: Annotated[int, typer.Option(help="Number of players to display.")] = 20,
    categories: Annotated[str | None, typer.Option(help="Comma-separated categories (e.g. hr,sb,obp).")] = None,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    method: Annotated[str, typer.Option(help="Valuation method to use.")] = DEFAULT_METHOD,
) -> None:
    """Compute player valuations from projections."""
    validate_engine(engine)
    validate_method(method)

    if year is None:
        year = datetime.now().year

    show_batting = not pitching or batting
    show_pitching = not batting or pitching

    container = get_container()
    pipeline = PIPELINES[engine]()

    league_settings = load_league_settings()

    if show_batting:
        batting_cats = (
            _parse_categories(categories, _SUPPORTED_BATTING) if categories else league_settings.batting_categories
        )
        batting_projections: list[BattingProjection] = pipeline.project_batters(container.batting_source, container.team_batting_source, year)
        batting_values: list[PlayerValue] = zscore_batting(batting_projections, batting_cats)
        batting_values.sort(key=lambda pv: pv.total_value, reverse=True)
        cat_label = ", ".join(c.value for c in batting_cats)
        title = f"Z-score batting values for {year} ({cat_label})"
        _print_value_table(title, batting_cats, batting_values, top)

    if show_batting and show_pitching:
        console.print()

    if show_pitching:
        pitching_cats = (
            _parse_categories(categories, _SUPPORTED_PITCHING) if categories else league_settings.pitching_categories
        )
        pitching_projections: list[PitchingProjection] = pipeline.project_pitchers(container.pitching_source, container.team_pitching_source, year)
        pitching_values: list[PlayerValue] = zscore_pitching(pitching_projections, pitching_cats)
        pitching_values.sort(key=lambda pv: pv.total_value, reverse=True)
        cat_label = ", ".join(c.value for c in pitching_cats)
        title = f"Z-score pitching values for {year} ({cat_label})"
        _print_value_table(title, pitching_cats, pitching_values, top)
