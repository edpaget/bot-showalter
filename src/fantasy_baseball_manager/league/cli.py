import logging
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, cast

import typer

from fantasy_baseball_manager.cache.factory import create_cache_store, get_cache_key
from fantasy_baseball_manager.cache.sources import CachedRosterSource
from fantasy_baseball_manager.config import AppConfig, apply_cli_overrides, clear_cli_overrides, create_config
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, DEFAULT_METHOD, validate_engine, validate_method
from fantasy_baseball_manager.league.models import TeamProjection
from fantasy_baseball_manager.league.projections import match_projections
from fantasy_baseball_manager.league.roster import RosterSource, YahooRosterSource
from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource, StatsDataSource
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper, build_cached_sfbb_mapper, build_sfbb_mapper
from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

logger = logging.getLogger(__name__)

COMPARE_SORT_FIELDS: dict[str, Callable[[TeamProjection], float]] = {
    "total_hr": lambda t: t.total_hr,
    "total_sb": lambda t: t.total_sb,
    "total_h": lambda t: t.total_h,
    "total_pa": lambda t: t.total_pa,
    "team_avg": lambda t: t.team_avg,
    "team_obp": lambda t: t.team_obp,
    "total_ip": lambda t: t.total_ip,
    "total_so": lambda t: t.total_so,
    "team_era": lambda t: -t.team_era,  # lower is better
    "team_whip": lambda t: -t.team_whip,  # lower is better
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
    config = cast("AppConfig", create_config())
    client = YahooFantasyClient(config)

    if target_season is None and config["league.is_keeper"]:
        current_league = client.get_league()
        draft_status = current_league.settings().get("draft_status", "")
        if draft_status == "predraft":
            target_season = int(str(config["league.season"])) - 1
            logger.debug("Keeper league in predraft — using previous season %d", target_season)

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
    config = cast("AppConfig", create_config())
    ttl = int(str(config["cache.id_mappings_ttl"]))
    cache_store = create_cache_store(config)
    cache_key = get_cache_key(config)
    return build_cached_sfbb_mapper(cache_store, cache_key, ttl)


def _get_data_source() -> StatsDataSource:
    if _data_source_factory is not None:
        return _data_source_factory()
    return PybaseballDataSource()


def format_team_projections(team_projections: list[TeamProjection], top: int, sort_by: str) -> str:
    lines: list[str] = []

    for team in team_projections:
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  {team.team_name} ({team.team_key})")
        lines.append(f"{'=' * 70}")

        batters = [p for p in team.players if p.roster_player.position_type == "B"]
        pitchers = [p for p in team.players if p.roster_player.position_type == "P"]

        if batters:
            lines.append(f"\n  {'Batters':<25} {'PA':>6} {'HR':>5} {'AVG':>6} {'OBP':>6} {'SB':>5}")
            lines.append(f"  {'-' * 56}")
            for pm in batters:
                if pm.batting_projection is not None:
                    bp = pm.batting_projection
                    avg = bp.h / bp.ab if bp.ab > 0 else 0
                    obp = (bp.h + bp.bb + bp.hbp) / bp.pa if bp.pa > 0 else 0
                    lines.append(
                        f"  {pm.roster_player.name:<25} {bp.pa:>6.0f} {bp.hr:>5.1f} {avg:>6.3f} {obp:>6.3f} {bp.sb:>5.1f}"
                    )
                else:
                    lines.append(f"  {pm.roster_player.name:<25} {'--':>6} {'--':>5} {'--':>6} {'--':>6} {'--':>5}")

        if pitchers:
            lines.append(f"\n  {'Pitchers':<25} {'IP':>6} {'ERA':>5} {'WHIP':>6} {'SO':>5}")
            lines.append(f"  {'-' * 50}")
            for pm in pitchers:
                if pm.pitching_projection is not None:
                    pp = pm.pitching_projection
                    lines.append(
                        f"  {pm.roster_player.name:<25} {pp.ip:>6.1f} {pp.era:>5.2f} {pp.whip:>6.3f} {pp.so:>5.1f}"
                    )
                else:
                    lines.append(f"  {pm.roster_player.name:<25} {'--':>6} {'--':>5} {'--':>6} {'--':>5}")

        if team.unmatched_count > 0:
            lines.append(f"\n  Warning: {team.unmatched_count} player(s) could not be matched to projections")

    return "\n".join(lines)


def format_compare_table(team_projections: list[TeamProjection]) -> str:
    lines: list[str] = []
    header = (
        f"{'Team':<25} {'HR':>5} {'SB':>5} {'AVG':>6} {'OBP':>6}"
        f" {'IP':>6} {'SO':>5} {'ERA':>5} {'WHIP':>5} {'?':>3}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for t in team_projections:
        lines.append(
            f"{t.team_name:<25} {t.total_hr:>5.0f} {t.total_sb:>5.0f} {t.team_avg:>6.3f} {t.team_obp:>6.3f}"
            f" {t.total_ip:>6.0f} {t.total_so:>5.0f} {t.team_era:>5.2f} {t.team_whip:>5.3f} {t.unmatched_count:>3}"
        )

    return "\n".join(lines)


def _invalidate_caches() -> None:
    """Invalidate all cached data so the next cached run fetches fresh."""
    cache_store = create_cache_store()
    cache_key = get_cache_key()
    for ns in ("rosters", "sfbb_csv"):
        cache_store.invalidate(ns, cache_key)
    logger.debug("Invalidated cached rosters and sfbb_csv for key=%s", cache_key)


def _load_team_projections(
    year: int, engine: str = DEFAULT_ENGINE, no_cache: bool = False
) -> list[TeamProjection]:
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
    top: Annotated[int, typer.Option(help="Number of players per team to display.")] = 25,
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

        typer.echo(f"League projections for {year}\n")

        team_projections = _load_team_projections(year, engine=engine, no_cache=no_cache)
        team_projections.sort(key=COMPARE_SORT_FIELDS[sort_by], reverse=True)

        typer.echo(format_team_projections(team_projections, top, sort_by))
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

        typer.echo(f"League comparison for {year}\n")

        team_projections = _load_team_projections(year, engine=engine, no_cache=no_cache)
        team_projections.sort(key=COMPARE_SORT_FIELDS[sort_by], reverse=True)

        typer.echo(format_compare_table(team_projections))
    finally:
        clear_cli_overrides()
