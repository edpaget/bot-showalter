from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml

from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot
from fantasy_baseball_manager.draft.positions import (
    DEFAULT_ROSTER_CONFIG,
    PositionSource,
    YahooPositionSource,
    infer_pitcher_role,
    load_positions_file,
)
from fantasy_baseball_manager.draft.state import DraftState
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.marcel.batting import project_batters
from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource, StatsDataSource
from fantasy_baseball_manager.marcel.pitching import project_pitchers
from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

_CATEGORY_MAP: dict[str, StatCategory] = {member.value.lower(): member for member in StatCategory}

_DEFAULT_BATTING_CATS: tuple[StatCategory, ...] = (StatCategory.HR, StatCategory.SB, StatCategory.OBP)
_DEFAULT_PITCHING_CATS: tuple[StatCategory, ...] = (StatCategory.K, StatCategory.ERA, StatCategory.WHIP)

# Module-level factories for dependency injection in tests
_data_source_factory: Callable[[], StatsDataSource] = PybaseballDataSource
_id_mapper_factory: Callable[[], PlayerIdMapper] | None = None
_yahoo_league_factory: Callable[[], object] | None = None


def set_data_source_factory(factory: Callable[[], StatsDataSource]) -> None:
    global _data_source_factory
    _data_source_factory = factory


def set_id_mapper_factory(factory: Callable[[], PlayerIdMapper]) -> None:
    global _id_mapper_factory
    _id_mapper_factory = factory


def set_yahoo_league_factory(factory: Callable[[], object]) -> None:
    global _yahoo_league_factory
    _yahoo_league_factory = factory


def _get_id_mapper() -> PlayerIdMapper:
    if _id_mapper_factory is not None:
        return _id_mapper_factory()
    from fantasy_baseball_manager.player_id.mapper import ChadwickMapper

    return ChadwickMapper()


def _get_yahoo_league() -> object:
    if _yahoo_league_factory is not None:
        return _yahoo_league_factory()
    from fantasy_baseball_manager.config import create_config
    from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

    config = create_config()
    client = YahooFantasyClient(config)  # type: ignore[arg-type]
    return client.get_league()


def _load_drafted_file(path: Path) -> set[str]:
    drafted: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        drafted.add(line)
    return drafted


def _load_my_picks_file(path: Path) -> list[tuple[str, str]]:
    picks: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", maxsplit=1)
        if len(parts) == 2:
            picks.append((parts[0].strip(), parts[1].strip()))
    return picks


def _parse_weight(raw: str) -> tuple[StatCategory, float]:
    parts = raw.split("=", maxsplit=1)
    if len(parts) != 2:
        typer.echo(f"Invalid weight format: {raw!r} (expected CAT=N)", err=True)
        raise typer.Exit(code=1)
    key = parts[0].strip().lower()
    if key not in _CATEGORY_MAP:
        typer.echo(f"Unknown category in weight: {parts[0].strip()}", err=True)
        raise typer.Exit(code=1)
    try:
        value = float(parts[1].strip())
    except ValueError as err:
        typer.echo(f"Invalid weight value: {parts[1].strip()!r}", err=True)
        raise typer.Exit(code=1) from err
    return _CATEGORY_MAP[key], value


def _load_roster_config(path: Path) -> RosterConfig:
    data = yaml.safe_load(path.read_text())
    slots_data = data.get("slots", {})
    slots: list[RosterSlot] = []
    for position, count in slots_data.items():
        slots.append(RosterSlot(position=str(position), count=int(count)))
    return RosterConfig(slots=tuple(slots))


def draft_rank(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    drafted: Annotated[
        Path | None, typer.Option("--drafted", help="File of drafted player IDs (one per line).")
    ] = None,
    my_picks: Annotated[
        Path | None, typer.Option("--my-picks", help="File of user picks (player_id,position per line).")
    ] = None,
    roster_config_file: Annotated[Path | None, typer.Option("--roster-config", help="YAML roster slot config.")] = None,
    positions_file: Annotated[
        Path | None, typer.Option("--positions", help="CSV of player_id,positions for positional need.")
    ] = None,
    yahoo_positions: Annotated[
        bool, typer.Option("--yahoo-positions", help="Fetch position eligibility from Yahoo Fantasy API.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data from Yahoo API.")
    ] = False,
    weight: Annotated[
        list[str] | None, typer.Option("--weight", help="Category weight multiplier (e.g. HR=2.0).")
    ] = None,
    top: Annotated[int, typer.Option(help="Number of players to show.")] = 50,
    batting: Annotated[bool, typer.Option("--batting", help="Show only batters.")] = False,
    pitching: Annotated[bool, typer.Option("--pitching", help="Show only pitchers.")] = False,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
) -> None:
    """Produce a ranked draft board from z-score valuations."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    show_batting = not pitching or batting
    show_pitching = not batting or pitching

    # Load roster config
    roster_config = _load_roster_config(roster_config_file) if roster_config_file else DEFAULT_ROSTER_CONFIG

    # Load position data
    player_positions: dict[str, tuple[str, ...]] = {}
    if positions_file:
        player_positions = load_positions_file(positions_file)
    elif yahoo_positions:
        league = _get_yahoo_league()
        id_mapper = _get_id_mapper()
        source: PositionSource = YahooPositionSource(league, id_mapper)  # type: ignore[arg-type]
        if not no_cache:
            from fantasy_baseball_manager.cache.sources import CachedPositionSource
            from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore
            from fantasy_baseball_manager.config import create_config

            config = create_config()
            db_path = Path(str(config["cache.db_path"])).expanduser()
            ttl = int(str(config["cache.positions_ttl"]))
            cache_key = f"{config['league.game_code']}_{config['league.id']}"
            cache_store = SqliteCacheStore(db_path)
            source = CachedPositionSource(source, cache_store, cache_key, ttl)
        player_positions = source.fetch_positions()

    # Parse category weights
    category_weights: dict[StatCategory, float] = {}
    if weight:
        for w in weight:
            cat, val = _parse_weight(w)
            category_weights[cat] = val

    # Generate projections and valuations
    data_source = _data_source_factory()
    all_values: list[PlayerValue] = []

    if show_batting:
        batting_projections = project_batters(data_source, year)
        batting_values = zscore_batting(batting_projections, _DEFAULT_BATTING_CATS)
        all_values.extend(batting_values)

    if show_pitching:
        pitching_projections = project_pitchers(data_source, year)
        # Infer pitcher positions
        for proj in pitching_projections:
            if proj.player_id not in player_positions:
                role = infer_pitcher_role(proj)
                player_positions[proj.player_id] = (role,)
        pitching_values = zscore_pitching(pitching_projections, _DEFAULT_PITCHING_CATS)
        all_values.extend(pitching_values)

    # Build draft state
    state = DraftState(
        roster_config=roster_config,
        player_values=all_values,
        player_positions=player_positions,
        category_weights=category_weights,
    )

    # Apply drafted players
    drafted_ids: set[str] = set()
    if drafted:
        drafted_ids = _load_drafted_file(drafted)
    my_pick_list: list[tuple[str, str]] = []
    if my_picks:
        my_pick_list = _load_my_picks_file(my_picks)

    for pid in drafted_ids:
        # Check if this is also a user pick (will be handled below)
        user_pick_ids = {p[0] for p in my_pick_list}
        if pid not in user_pick_ids:
            state.draft_player(pid, is_user=False)

    for pid, pos in my_pick_list:
        state.draft_player(pid, is_user=True, position=pos)

    # Get and display rankings
    rankings = state.get_rankings(limit=top)

    if not rankings:
        typer.echo("No players to rank.")
        return

    # Build output table
    lines: list[str] = []
    lines.append(f"Draft rankings for {year}:")

    header = f"{'Rk':>4} {'Name':<25} {'Pos':<8} {'Best':>4} {'Mult':>5} {'Raw':>7} {'Wtd':>7} {'Adj':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in rankings:
        pos_str = "/".join(r.eligible_positions) if r.eligible_positions else "-"
        best_str = r.best_position or "-"
        lines.append(
            f"{r.rank:>4} {r.name:<25} {pos_str:<8} {best_str:>4} {r.position_multiplier:>5.2f}"
            f" {r.raw_value:>7.1f} {r.weighted_value:>7.1f} {r.adjusted_value:>7.1f}"
        )

    typer.echo("\n".join(lines))
