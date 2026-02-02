import logging
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, cast

import typer

from fantasy_baseball_manager.cache.factory import create_cache_store, get_cache_key
from fantasy_baseball_manager.cache.sources import CachedRosterSource
from fantasy_baseball_manager.config import (
    AppConfig,
    apply_cli_overrides,
    clear_cli_overrides,
    create_config,
    load_league_settings,
)
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, DEFAULT_METHOD, validate_engine, validate_method
from fantasy_baseball_manager.league.models import TeamProjection
from fantasy_baseball_manager.league.projections import match_projections
from fantasy_baseball_manager.league.roster import RosterSource, YahooRosterSource
from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource, StatsDataSource
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper, build_cached_sfbb_mapper, build_sfbb_mapper
from fantasy_baseball_manager.valuation.models import LeagueSettings, StatCategory
from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

logger = logging.getLogger(__name__)

COMPARE_SORT_FIELDS: dict[str, Callable[[TeamProjection], float]] = {
    "total_hr": lambda t: t.total_hr,
    "total_sb": lambda t: t.total_sb,
    "total_h": lambda t: t.total_h,
    "total_pa": lambda t: t.total_pa,
    "team_avg": lambda t: t.team_avg,
    "team_obp": lambda t: t.team_obp,
    "total_r": lambda t: t.total_r,
    "total_rbi": lambda t: t.total_rbi,
    "total_ip": lambda t: t.total_ip,
    "total_so": lambda t: t.total_so,
    "total_w": lambda t: t.total_w,
    "total_nsvh": lambda t: t.total_nsvh,
    "team_era": lambda t: -t.team_era,  # lower is better
    "team_whip": lambda t: -t.team_whip,  # lower is better
}

# Per-player column specs: (header, width, format_spec, extractor)
_BATTER_COLUMNS: dict[StatCategory, tuple[str, int, str, Callable[[BattingProjection], float]]] = {
    StatCategory.HR: ("HR", 5, "5.1f", lambda bp: bp.hr),
    StatCategory.R: ("R", 5, "5.1f", lambda bp: bp.r),
    StatCategory.RBI: ("RBI", 5, "5.1f", lambda bp: bp.rbi),
    StatCategory.SB: ("SB", 5, "5.1f", lambda bp: bp.sb),
    StatCategory.OBP: ("OBP", 6, "6.3f", lambda bp: (bp.h + bp.bb + bp.hbp) / bp.pa if bp.pa > 0 else 0),
}

_PITCHER_COLUMNS: dict[StatCategory, tuple[str, int, str, Callable[[PitchingProjection], float]]] = {
    StatCategory.W: ("W", 5, "5.1f", lambda pp: pp.w),
    StatCategory.K: ("K", 5, "5.1f", lambda pp: pp.so),
    StatCategory.ERA: ("ERA", 5, "5.2f", lambda pp: pp.era),
    StatCategory.WHIP: ("WHIP", 6, "6.3f", lambda pp: pp.whip),
    StatCategory.NSVH: ("NSVH", 5, "5.1f", lambda pp: pp.nsvh),
}

# Team aggregate column specs: (header, width, format_spec, extractor)
_TEAM_BATTING_COLUMNS: dict[StatCategory, tuple[str, int, str, Callable[[TeamProjection], float]]] = {
    StatCategory.HR: ("HR", 5, "5.0f", lambda t: t.total_hr),
    StatCategory.R: ("R", 5, "5.0f", lambda t: t.total_r),
    StatCategory.RBI: ("RBI", 5, "5.0f", lambda t: t.total_rbi),
    StatCategory.SB: ("SB", 5, "5.0f", lambda t: t.total_sb),
    StatCategory.OBP: ("OBP", 6, "6.3f", lambda t: t.team_obp),
}

_TEAM_PITCHING_COLUMNS: dict[StatCategory, tuple[str, int, str, Callable[[TeamProjection], float]]] = {
    StatCategory.W: ("W", 5, "5.0f", lambda t: t.total_w),
    StatCategory.K: ("K", 5, "5.0f", lambda t: t.total_so),
    StatCategory.ERA: ("ERA", 5, "5.2f", lambda t: t.team_era),
    StatCategory.WHIP: ("WHIP", 5, "5.3f", lambda t: t.team_whip),
    StatCategory.NSVH: ("NSVH", 5, "5.0f", lambda t: t.total_nsvh),
}

# Module-level DI factories for testing
_roster_source_factory: Callable[[], RosterSource] | None = None
_id_mapper_factory: Callable[[], PlayerIdMapper] | None = None
_data_source_factory: Callable[[], StatsDataSource] | None = None


def set_roster_source_factory(factory: Callable[[], RosterSource]) -> None:
    global _roster_source_factory
    _roster_source_factory = factory


def set_id_mapper_factory(factory: Callable[[], PlayerIdMapper]) -> None:
    global _id_mapper_factory
    _id_mapper_factory = factory


def set_data_source_factory(factory: Callable[[], StatsDataSource]) -> None:
    global _data_source_factory
    _data_source_factory = factory


def _get_roster_source(no_cache: bool = False, target_season: int | None = None) -> RosterSource:
    if _roster_source_factory is not None:
        return _roster_source_factory()
    config = create_config()
    client = YahooFantasyClient(cast("AppConfig", config))

    if target_season is None and config["league.is_keeper"]:
        current_league = client.get_league()
        draft_status = current_league.settings().get("draft_status", "")
        logger.debug("Keeper league draft_status=%r", draft_status)
        if draft_status == "predraft":
            target_season = int(str(config["league.season"])) - 1
            logger.debug("Using previous season %d for roster source", target_season)

    league = client.get_league_for_season(target_season) if target_season is not None else client.get_league()
    source: RosterSource = YahooRosterSource(league)
    if not no_cache:
        ttl = int(str(config["cache.rosters_ttl"]))
        cache_store = create_cache_store(config)
        cache_key = get_cache_key(config)
        source = CachedRosterSource(source, cache_store, cache_key, ttl)
    return source


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


def _get_data_source() -> StatsDataSource:
    if _data_source_factory is not None:
        return _data_source_factory()
    return PybaseballDataSource()


def format_team_projections(
    team_projections: list[TeamProjection],
    league_settings: LeagueSettings,
) -> str:
    bat_cols = [(cat, _BATTER_COLUMNS[cat]) for cat in league_settings.batting_categories if cat in _BATTER_COLUMNS]
    pit_cols = [(cat, _PITCHER_COLUMNS[cat]) for cat in league_settings.pitching_categories if cat in _PITCHER_COLUMNS]

    lines: list[str] = []

    for team in team_projections:
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  {team.team_name} ({team.team_key})")
        lines.append(f"{'=' * 70}")

        batters = [p for p in team.players if p.roster_player.position_type == "B"]
        pitchers = [p for p in team.players if p.roster_player.position_type == "P"]

        if batters:
            hdr = f"  {'Batters':<25} {'PA':>6}"
            for _, (header, width, _, _) in bat_cols:
                hdr += f" {header:>{width}}"
            lines.append(f"\n{hdr}")
            lines.append(f"  {'-' * (len(hdr) - 2)}")
            for pm in batters:
                if pm.batting_projection is not None:
                    bp = pm.batting_projection
                    row = f"  {pm.roster_player.name:<25} {bp.pa:>6.0f}"
                    for _, (_, _width, fmt, extract) in bat_cols:
                        row += f" {extract(bp):>{fmt}}"
                    lines.append(row)
                else:
                    row = f"  {pm.roster_player.name:<25} {'--':>6}"
                    for _, (_, width, _, _) in bat_cols:
                        row += f" {'--':>{width}}"
                    lines.append(row)

        if pitchers:
            hdr = f"  {'Pitchers':<25} {'IP':>6}"
            for _, (header, width, _, _) in pit_cols:
                hdr += f" {header:>{width}}"
            lines.append(f"\n{hdr}")
            lines.append(f"  {'-' * (len(hdr) - 2)}")
            for pm in pitchers:
                if pm.pitching_projection is not None:
                    pp = pm.pitching_projection
                    row = f"  {pm.roster_player.name:<25} {pp.ip:>6.1f}"
                    for _, (_, _width, fmt, extract) in pit_cols:
                        row += f" {extract(pp):>{fmt}}"
                    lines.append(row)
                else:
                    row = f"  {pm.roster_player.name:<25} {'--':>6}"
                    for _, (_, width, _, _) in pit_cols:
                        row += f" {'--':>{width}}"
                    lines.append(row)

        if team.unmatched_count > 0:
            lines.append(f"\n  Warning: {team.unmatched_count} player(s) could not be matched to projections")

    return "\n".join(lines)


def format_compare_table(
    team_projections: list[TeamProjection],
    league_settings: LeagueSettings,
) -> str:
    bat_cols = [
        (cat, _TEAM_BATTING_COLUMNS[cat]) for cat in league_settings.batting_categories if cat in _TEAM_BATTING_COLUMNS
    ]
    pit_cols = [
        (cat, _TEAM_PITCHING_COLUMNS[cat])
        for cat in league_settings.pitching_categories
        if cat in _TEAM_PITCHING_COLUMNS
    ]

    lines: list[str] = []
    header = f"{'Team':<25}"
    for _, (hdr, width, _, _) in bat_cols:
        header += f" {hdr:>{width}}"
    for _, (hdr, width, _, _) in pit_cols:
        header += f" {hdr:>{width}}"
    header += f" {'?':>3}"
    lines.append(header)
    lines.append("-" * len(header))

    for t in team_projections:
        row = f"{t.team_name:<25}"
        for _, (_, _width, fmt, extract) in bat_cols:
            row += f" {extract(t):>{fmt}}"
        for _, (_, _width, fmt, extract) in pit_cols:
            row += f" {extract(t):>{fmt}}"
        row += f" {t.unmatched_count:>3}"
        lines.append(row)

    return "\n".join(lines)


def _invalidate_caches() -> None:
    """Invalidate all cached data so the next cached run fetches fresh."""
    cache_store = create_cache_store()
    cache_key = get_cache_key()
    for ns in ("rosters", "sfbb_csv"):
        cache_store.invalidate(ns, cache_key)
    logger.debug("Invalidated cached rosters and sfbb_csv for key=%s", cache_key)


def _load_team_projections(year: int, engine: str = DEFAULT_ENGINE, no_cache: bool = False) -> list[TeamProjection]:
    if no_cache:
        _invalidate_caches()
    roster_source = _get_roster_source(no_cache=no_cache)
    data_source = _get_data_source()

    pipeline = PIPELINES[engine]()
    rosters = roster_source.fetch_rosters()

    id_mapper = _get_id_mapper(no_cache=no_cache)
    batting = pipeline.project_batters(data_source, year)
    pitching = pipeline.project_pitchers(data_source, year)

    return match_projections(rosters, batting, pitching, id_mapper)


def projections(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    sort_by: Annotated[str, typer.Option(help="Stat to sort teams by.")] = "total_hr",
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data from Yahoo API.")
    ] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Show projections for all rostered players in the league."""
    apply_cli_overrides(league_id, season)
    try:
        validate_engine(engine)

        if year is None:
            year = datetime.now().year

        if sort_by not in COMPARE_SORT_FIELDS:
            typer.echo(f"Unknown sort field: {sort_by}", err=True)
            raise typer.Exit(code=1)

        league_settings = load_league_settings()
        typer.echo(f"League projections for {year}\n")

        team_projections = _load_team_projections(year, engine=engine, no_cache=no_cache)
        team_projections.sort(key=COMPARE_SORT_FIELDS[sort_by], reverse=True)

        typer.echo(format_team_projections(team_projections, league_settings))
    finally:
        clear_cli_overrides()


def compare(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    sort_by: Annotated[str, typer.Option(help="Stat to sort by.")] = "total_hr",
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    method: Annotated[str, typer.Option(help="Valuation method to use.")] = DEFAULT_METHOD,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data from Yahoo API.")
    ] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Compare aggregate projected stats across all teams in the league."""
    apply_cli_overrides(league_id, season)
    try:
        validate_engine(engine)
        validate_method(method)

        if year is None:
            year = datetime.now().year

        if sort_by not in COMPARE_SORT_FIELDS:
            typer.echo(f"Unknown sort field: {sort_by}", err=True)
            raise typer.Exit(code=1)

        league_settings = load_league_settings()
        typer.echo(f"League comparison for {year}\n")

        team_projections = _load_team_projections(year, engine=engine, no_cache=no_cache)
        team_projections.sort(key=COMPARE_SORT_FIELDS[sort_by], reverse=True)

        typer.echo(format_compare_table(team_projections, league_settings))
    finally:
        clear_cli_overrides()
