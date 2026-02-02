from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime by typer
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from collections.abc import Callable

import typer
import yaml

from fantasy_baseball_manager.cache.factory import create_cache_store, get_cache_key
from fantasy_baseball_manager.cache.sources import CachedDraftResultsSource, CachedPositionSource
from fantasy_baseball_manager.config import (
    apply_cli_overrides,
    clear_cli_overrides,
    create_config,
    load_league_settings,
)
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
from fantasy_baseball_manager.draft.simulation_report import format_pick_log, format_standings
from fantasy_baseball_manager.draft.state import DraftState
from fantasy_baseball_manager.draft.strategy_presets import STRATEGY_PRESETS
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource, StatsDataSource
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper, build_cached_sfbb_mapper, build_sfbb_mapper
from fantasy_baseball_manager.valuation.models import PlayerValue, StatCategory
from fantasy_baseball_manager.valuation.zscore import zscore_batting, zscore_pitching
from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

logger = logging.getLogger(__name__)

_CATEGORY_MAP: dict[str, StatCategory] = {member.value.lower(): member for member in StatCategory}

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


def _get_id_mapper(no_cache: bool = False) -> PlayerIdMapper:
    if _id_mapper_factory is not None:
        return _id_mapper_factory()
    if no_cache:
        return build_sfbb_mapper()
    config = create_config()
    ttl = int(str(config["cache.id_mappings_ttl"]))
    cache_store = create_cache_store(config)
    cache_key = get_cache_key(config)
    return build_cached_sfbb_mapper(cache_store, cache_key, ttl)


def _get_yahoo_league() -> object:
    if _yahoo_league_factory is not None:
        return _yahoo_league_factory()
    config = create_config()
    client = YahooFantasyClient(config)  # type: ignore[arg-type]
    return client.get_league()


def _invalidate_caches() -> None:
    """Invalidate all cached data so the next cached run fetches fresh."""
    cache_store = create_cache_store()
    cache_key = get_cache_key()
    for ns in ("positions", "sfbb_csv", "draft_results"):
        cache_store.invalidate(ns, cache_key)
    logger.debug("Invalidated cached positions, sfbb_csv, and draft_results for key=%s", cache_key)


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
    apply_cli_overrides(league_id, season)
    try:
        validate_engine(engine)

        if year is None:
            year = datetime.now().year

        show_batting = not pitching or batting
        show_pitching = not batting or pitching

        # Load roster config
        roster_config = _load_roster_config(roster_config_file) if roster_config_file else DEFAULT_ROSTER_CONFIG

        # Load position data
        player_positions: dict[str, tuple[str, ...]] = {}
        yahoo_league: object | None = None
        yahoo_id_mapper: PlayerIdMapper | None = None
        if yahoo:
            yahoo_league = _get_yahoo_league()
            yahoo_id_mapper = _get_id_mapper(no_cache=no_cache)

        if positions_file:
            player_positions = load_positions_file(positions_file)
        elif yahoo and yahoo_league is not None and yahoo_id_mapper is not None:
            source: PositionSource = YahooPositionSource(yahoo_league, yahoo_id_mapper)  # type: ignore[arg-type]
            if not no_cache:
                config = create_config()
                ttl = int(str(config["cache.positions_ttl"]))
                cache_store = create_cache_store(config)
                cache_key = get_cache_key(config)
                source = CachedPositionSource(source, cache_store, cache_key, ttl)
            else:
                _invalidate_caches()
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
        data_source = _data_source_factory()
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
        if yahoo and yahoo_league is not None and yahoo_id_mapper is not None:
            draft_source = YahooDraftResultsSource(yahoo_league)
            draft_status = draft_source.fetch_draft_status()
            if not no_cache and draft_status != DraftStatus.IN_PROGRESS:
                config = create_config()
                dr_ttl = int(str(config["cache.draft_results_ttl"]))
                cache_store = create_cache_store(config)
                cache_key = get_cache_key(config)
                draft_source = CachedDraftResultsSource(draft_source, cache_store, cache_key, dr_ttl)  # type: ignore[assignment]
            picks = draft_source.fetch_draft_results()
            user_team_key = draft_source.fetch_user_team_key()
            logger.debug("User team key: %s, draft picks: %d", user_team_key, len(picks))
            for pick in picks:
                fg_id = yahoo_id_mapper.yahoo_to_fangraphs(pick.player_id)
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
        lines: list[str] = []
        lines.append(f"Draft rankings for {year}:")

        header = f"{'Rk':>4} {'Name':<25} {'Pos':<8} {'Mult':>5} {'Raw':>7} {'Wtd':>7} {'Adj':>7}"
        lines.append(header)
        lines.append("-" * len(header))

        for r in rankings:
            display_pos = tuple(p for p in r.eligible_positions if p != "Util") or r.eligible_positions
            pos_str = "/".join(display_pos) if display_pos else "-"
            lines.append(
                f"{r.rank:>4} {r.name:<25} {pos_str:<8}"
                f" {r.position_multiplier:>5.2f}"
                f" {r.raw_value:>7.1f} {r.weighted_value:>7.1f} {r.adjusted_value:>7.1f}"
            )

        typer.echo("\n".join(lines))
    finally:
        clear_cli_overrides()


def _build_projections_and_positions(
    engine: str,
    year: int,
) -> tuple[list[PlayerValue], dict[tuple[str, str], tuple[str, ...]]]:
    """Build player values and composite positions for simulation."""
    league_settings = load_league_settings()
    data_source = _data_source_factory()
    pipeline = PIPELINES[engine]()

    batting_projections = pipeline.project_batters(data_source, year)
    batting_values = zscore_batting(batting_projections, league_settings.batting_categories)
    batting_ids = {p.player_id for p in batting_projections}

    pitching_projections = pipeline.project_pitchers(data_source, year)
    player_positions: dict[str, tuple[str, ...]] = {}
    for proj in pitching_projections:
        if proj.player_id not in player_positions:
            role = infer_pitcher_role(proj)
            player_positions[proj.player_id] = (role,)
    pitching_values = zscore_pitching(pitching_projections, league_settings.pitching_categories)
    pitching_ids = {p.player_id for p in pitching_projections}

    all_values: list[PlayerValue] = list(batting_values) + list(pitching_values)

    # Build composite-keyed positions dict
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
            composite_positions[(pid, "B")] = positions
            composite_positions[(pid, "P")] = positions

    return all_values, composite_positions


def _load_keepers_file(path: Path) -> dict[int, dict[str, object]]:
    """Load keepers YAML file. Returns dict mapping team_id to team config."""
    data = yaml.safe_load(path.read_text())
    teams_raw = data.get("teams", {})
    teams_data: dict[int, dict[str, object]] = {}
    if isinstance(teams_raw, dict):
        for team_id, team_info in teams_raw.items():
            teams_data[int(team_id)] = dict(team_info)  # type: ignore[arg-type]
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
        keepers_list: list[str] = list(keepers_raw) if isinstance(keepers_raw, (list, tuple)) else []
        keeper_ids: tuple[str, ...] = tuple(str(k) for k in keepers_list)

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
    all_values, composite_positions = _build_projections_and_positions(engine, year)
    typer.echo(f"Running {teams}-team, {rounds}-round snake draft simulation...")

    result = simulate_draft(sim_config, all_values, composite_positions)

    # Output
    if log:
        typer.echo(format_pick_log(result))
        typer.echo("")

    if rosters:
        from fantasy_baseball_manager.draft.simulation_report import format_team_roster

        typer.echo("Team Rosters")
        typer.echo("=" * 80)
        for tr in result.team_results:
            typer.echo(format_team_roster(tr, result.pick_log))
        typer.echo("")

    if standings:
        typer.echo(format_standings(result))
