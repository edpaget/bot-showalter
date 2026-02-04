from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime by typer
from typing import TYPE_CHECKING, Annotated, cast

if TYPE_CHECKING:
    import yahoo_fantasy_api

import typer
import yaml
from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.cache.sources import CachedDraftResultsSource, CachedPositionSource
from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.draft.models import RosterConfig, RosterSlot
from fantasy_baseball_manager.draft.positions import (
    DEFAULT_ROSTER_CONFIG,
    PositionSource,
    YahooPositionSource,
    infer_pitcher_role,
    load_positions_file,
)
from fantasy_baseball_manager.draft.results import DraftStatus, YahooDraftResultsSource
from fantasy_baseball_manager.draft.simulation import simulate_draft
from fantasy_baseball_manager.draft.simulation_models import (
    SimulationConfig,
    TeamConfig,
)
from fantasy_baseball_manager.draft.simulation_report import print_pick_log, print_standings, print_team_roster
from fantasy_baseball_manager.draft.state import DraftState
from fantasy_baseball_manager.draft.strategy_presets import STRATEGY_PRESETS
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.services import cli_context, get_container, set_container
from fantasy_baseball_manager.shared.orchestration import (
    build_projections_and_positions,
)
from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching

logger = logging.getLogger(__name__)

console = Console()

__all__ = ["build_projections_and_positions", "draft_rank", "draft_simulate", "set_container"]

_CATEGORY_MAP: dict[str, StatCategory] = {member.value.lower(): member for member in StatCategory}


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
    roster_config_file: Annotated[Path | None, typer.Option("--roster-config", help="YAML roster slot config.")] = None,
    positions_file: Annotated[
        Path | None, typer.Option("--positions", help="CSV of player_id,positions for positional need.")
    ] = None,
    yahoo: Annotated[
        bool, typer.Option("--yahoo", help="Fetch positions and draft results from Yahoo Fantasy API.")
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
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Produce a ranked draft board from z-score valuations."""
    with cli_context(league_id=league_id, season=season, no_cache=no_cache):
        validate_engine(engine)

        if year is None:
            year = datetime.now().year

        show_batting = not pitching or batting
        show_pitching = not batting or pitching

        # Load roster config
        roster_config = _load_roster_config(roster_config_file) if roster_config_file else DEFAULT_ROSTER_CONFIG

        # Load position data
        container = get_container()
        player_positions: dict[str, tuple[str, ...]] = {}

        if positions_file:
            player_positions = load_positions_file(positions_file)
        elif yahoo:
            league = cast("yahoo_fantasy_api.League", container.yahoo_league)
            source: PositionSource = YahooPositionSource(league, container.id_mapper)
            if not container.config.no_cache:
                ttl = int(str(container.app_config["cache.positions_ttl"]))
                source = CachedPositionSource(source, container.cache_store, container.cache_key, ttl)
            else:
                container.invalidate_caches(("positions", "sfbb_csv", "draft_results"))
            player_positions = source.fetch_positions()

        logger.debug("Loaded %d player positions", len(player_positions))
        if player_positions:
            sample = list(player_positions.items())[:5]
            for pid, pos in sample:
                logger.debug("  position sample: %s -> %s", pid, pos)

        # Parse category weights
        category_weights: dict[StatCategory, float] = {}
        if weight:
            for w in weight:
                cat, val = _parse_weight(w)
                category_weights[cat] = val

        # Generate projections and valuations
        league_settings = load_league_settings()
        data_source = get_container().data_source
        pipeline = PIPELINES[engine]()
        all_values: list[PlayerValue] = []
        batting_ids: set[str] = set()
        pitching_ids: set[str] = set()

        if show_batting:
            batting_projections = pipeline.project_batters(data_source, year)
            batting_values = zscore_batting(batting_projections, league_settings.batting_categories)
            all_values.extend(batting_values)
            batting_ids = {p.player_id for p in batting_projections}
            logger.debug("Batting projections: %d players", len(batting_ids))
            if batting_ids:
                logger.debug("  sample batting IDs: %s", list(batting_ids)[:5])

        if show_pitching:
            pitching_projections = pipeline.project_pitchers(data_source, year)
            # Infer pitcher positions (into the plain player_positions dict)
            for proj in pitching_projections:
                if proj.player_id not in player_positions:
                    role = infer_pitcher_role(proj)
                    player_positions[proj.player_id] = (role,)
            pitching_values = zscore_pitching(pitching_projections, league_settings.pitching_categories)
            all_values.extend(pitching_values)
            pitching_ids = {p.player_id for p in pitching_projections}
            logger.debug("Pitching projections: %d players", len(pitching_ids))
            if pitching_ids:
                logger.debug("  sample pitching IDs: %s", list(pitching_ids)[:5])

        # Build composite-keyed positions dict for DraftState
        _PITCHER_POSITIONS: frozenset[str] = frozenset({"SP", "RP", "P"})
        two_way_ids = batting_ids & pitching_ids
        composite_positions: dict[tuple[str, str], tuple[str, ...]] = {}
        for pid, positions in player_positions.items():
            if pid in two_way_ids:
                batting_pos = tuple(p for p in positions if p not in _PITCHER_POSITIONS)
                pitching_pos = tuple(p for p in positions if p in _PITCHER_POSITIONS)
                if batting_pos:
                    composite_positions[(pid, "B")] = batting_pos
                if pitching_pos:
                    composite_positions[(pid, "P")] = pitching_pos
            elif pid in batting_ids:
                composite_positions[(pid, "B")] = positions
            elif pid in pitching_ids:
                composite_positions[(pid, "P")] = positions
            else:
                # Player not in projections — assign to both types if present
                composite_positions[(pid, "B")] = positions
                composite_positions[(pid, "P")] = positions

        position_ids_in_batting = {pid for pid, _ in composite_positions if pid in batting_ids}
        position_ids_in_pitching = {pid for pid, _ in composite_positions if pid in pitching_ids}
        logger.debug(
            "Composite positions: %d entries, %d match batting projections, %d match pitching projections",
            len(composite_positions),
            len(position_ids_in_batting),
            len(position_ids_in_pitching),
        )
        logger.debug("Total player values: %d", len(all_values))

        # Build draft state
        state = DraftState(
            roster_config=roster_config,
            player_values=all_values,
            player_positions=composite_positions,
            category_weights=category_weights,
        )

        # Apply draft results from Yahoo
        if yahoo:
            draft_league = cast("yahoo_fantasy_api.League", container.yahoo_league)
            draft_source = YahooDraftResultsSource(draft_league)
            draft_status = draft_source.fetch_draft_status()
            if not container.config.no_cache and draft_status != DraftStatus.IN_PROGRESS:
                dr_ttl = int(str(container.app_config["cache.draft_results_ttl"]))
                draft_source = CachedDraftResultsSource(
                    draft_source, container.cache_store, container.cache_key, dr_ttl
                )
            picks = draft_source.fetch_draft_results()
            user_team_key = draft_source.fetch_user_team_key()
            logger.debug("User team key: %s, draft picks: %d", user_team_key, len(picks))
            for pick in picks:
                fg_id = container.id_mapper.yahoo_to_fangraphs(pick.player_id)
                if fg_id is None:
                    logger.debug("No FanGraphs ID for Yahoo player %s, skipping", pick.player_id)
                    continue
                is_user = pick.team_key == user_team_key
                state.draft_player(fg_id, is_user=is_user)

        # Get and display rankings
        rankings = state.get_rankings(limit=top)

        if not rankings:
            typer.echo("No players to rank.")
            return

        # Build output table
        console.print(f"[bold]Draft rankings for {year}:[/bold]\n")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Rk", justify="right")
        table.add_column("Name")
        table.add_column("Pos")
        table.add_column("Mult", justify="right")
        table.add_column("Raw", justify="right")
        table.add_column("Wtd", justify="right")
        table.add_column("Adj", justify="right")

        for r in rankings:
            display_pos = tuple(p for p in r.eligible_positions if p != "Util") or r.eligible_positions
            pos_str = "/".join(display_pos) if display_pos else "-"
            table.add_row(
                str(r.rank),
                r.name,
                pos_str,
                f"{r.position_multiplier:.2f}",
                f"{r.raw_value:.1f}",
                f"{r.weighted_value:.1f}",
                f"{r.adjusted_value:.1f}",
            )

        console.print(table)


def _load_keepers_file(path: Path) -> dict[int, dict[str, object]]:
    """Load keepers YAML file. Returns dict mapping team_id to team config."""
    data = yaml.safe_load(path.read_text())
    teams_raw = data.get("teams", {})
    teams_data: dict[int, dict[str, object]] = {}
    if isinstance(teams_raw, dict):
        for team_id, team_info in teams_raw.items():
            teams_data[int(team_id)] = dict(team_info)
    return teams_data


def draft_simulate(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    user_pick: Annotated[int, typer.Option("--user-pick", help="User's draft position (1-based).")] = 1,
    user_strategy: Annotated[
        str, typer.Option("--user-strategy", help=f"Strategy for user team. Options: {', '.join(STRATEGY_PRESETS)}")
    ] = "balanced",
    opponent_strategy: Annotated[
        str, typer.Option("--opponent-strategy", help="Default strategy for opponent teams.")
    ] = "balanced",
    keepers_file: Annotated[
        Path | None, typer.Option("--keepers", help="YAML file with per-team keepers and optional strategies.")
    ] = None,
    roster_config_file: Annotated[Path | None, typer.Option("--roster-config", help="YAML roster slot config.")] = None,
    rounds: Annotated[int, typer.Option("--rounds", help="Total number of draft rounds.")] = 20,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed for reproducibility.")] = None,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    log: Annotated[bool, typer.Option("--log", help="Show pick-by-pick draft log.")] = False,
    rosters: Annotated[bool, typer.Option("--rosters/--no-rosters", help="Show final team rosters.")] = True,
    standings: Annotated[bool, typer.Option("--standings/--no-standings", help="Show projected standings.")] = True,
) -> None:
    """Simulate a full snake draft with configurable team strategies."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    if user_strategy not in STRATEGY_PRESETS:
        typer.echo(f"Unknown strategy: {user_strategy!r}. Options: {', '.join(STRATEGY_PRESETS)}", err=True)
        raise typer.Exit(code=1)
    if opponent_strategy not in STRATEGY_PRESETS:
        typer.echo(f"Unknown strategy: {opponent_strategy!r}. Options: {', '.join(STRATEGY_PRESETS)}", err=True)
        raise typer.Exit(code=1)

    roster_config = _load_roster_config(roster_config_file) if roster_config_file else DEFAULT_ROSTER_CONFIG

    # Load keepers file if provided
    keepers_data: dict[int, dict[str, object]] = {}
    if keepers_file:
        keepers_data = _load_keepers_file(keepers_file)

    # Build team configs
    team_configs: list[TeamConfig] = []
    for i in range(1, teams + 1):
        kd = keepers_data.get(i, {})
        name = str(kd.get("name", f"Team {i}"))
        keepers_raw = kd.get("keepers", [])
        keeper_ids: tuple[str, ...] = (
            tuple(str(k) for k in keepers_raw) if isinstance(keepers_raw, (list, tuple)) else ()
        )

        if i == user_pick:
            strategy = STRATEGY_PRESETS[user_strategy]
        else:
            team_strat_name = str(kd.get("strategy", opponent_strategy))
            if team_strat_name not in STRATEGY_PRESETS:
                typer.echo(
                    f"Unknown strategy for team {i}: {team_strat_name!r}. Options: {', '.join(STRATEGY_PRESETS)}",
                    err=True,
                )
                raise typer.Exit(code=1)
            strategy = STRATEGY_PRESETS[team_strat_name]

        team_configs.append(
            TeamConfig(
                team_id=i,
                name=name,
                strategy=strategy,
                keepers=keeper_ids,
            )
        )

    sim_config = SimulationConfig(
        teams=tuple(team_configs),
        roster_config=roster_config,
        total_rounds=rounds,
        seed=seed,
    )

    # Build projections
    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)
    typer.echo(f"Running {teams}-team, {rounds}-round snake draft simulation...")

    result = simulate_draft(sim_config, all_values, composite_positions)

    # Output
    if log:
        print_pick_log(result)
        console.print()

    if rosters:
        console.print("[bold]Team Rosters[/bold]")
        for tr in result.team_results:
            print_team_roster(tr)
            console.print()

    if standings:
        print_standings(result)
