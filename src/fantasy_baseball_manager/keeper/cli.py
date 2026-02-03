from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path  # noqa: TC003 — used at runtime by typer
from typing import TYPE_CHECKING, Annotated, cast

import typer
import yaml

from fantasy_baseball_manager.cache.factory import create_cache_store, get_cache_key
from fantasy_baseball_manager.cache.sources import CachedRosterSource
from fantasy_baseball_manager.config import (
    AppConfig,
    apply_cli_overrides,
    clear_cli_overrides,
    create_config,
)
from fantasy_baseball_manager.draft.cli import build_projections_and_positions
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.keeper.models import KeeperCandidate
from fantasy_baseball_manager.keeper.replacement import DraftPoolReplacementCalculator
from fantasy_baseball_manager.keeper.surplus import SurplusCalculator
from fantasy_baseball_manager.keeper.yahoo_source import YahooKeeperSource
from fantasy_baseball_manager.league.roster import RosterSource, YahooRosterSource
from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper, build_cached_sfbb_mapper, build_sfbb_mapper
from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.valuation.models import PlayerValue

logger = logging.getLogger(__name__)

keeper_app = typer.Typer(help="Keeper analysis commands.")

# Module-level DI factories for testing
_roster_source_factory: Callable[[], RosterSource] | None = None
_id_mapper_factory: Callable[[], PlayerIdMapper] | None = None
_yahoo_league_factory: Callable[[], object] | None = None


def set_roster_source_factory(factory: Callable[[], RosterSource]) -> None:
    global _roster_source_factory
    _roster_source_factory = factory


def set_id_mapper_factory(factory: Callable[[], PlayerIdMapper]) -> None:
    global _id_mapper_factory
    _id_mapper_factory = factory


def set_yahoo_league_factory(factory: Callable[[], object]) -> None:
    global _yahoo_league_factory
    _yahoo_league_factory = factory


def _get_roster_source_and_league(no_cache: bool = False) -> tuple[RosterSource, object]:
    """Build a roster source and return the league it was built from.

    For keeper leagues in predraft, both the roster source and league resolve
    to the previous season so that ``league.team_key()`` matches the roster
    team keys.
    """
    if _roster_source_factory is not None:
        # In test mode the factories are wired independently.
        league = _yahoo_league_factory() if _yahoo_league_factory is not None else object()
        return _roster_source_factory(), league

    config = create_config()
    client = YahooFantasyClient(cast("AppConfig", config))

    target_season: int | None = None
    if config["league.is_keeper"]:
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
    return source, league


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


def _load_keepers_file(path: Path) -> set[str]:
    """Load keepers YAML and return a flat set of all other teams' keeper player IDs."""
    data = yaml.safe_load(path.read_text())
    teams_raw = data.get("teams", {})
    keeper_ids: set[str] = set()
    if isinstance(teams_raw, dict):
        for team_info in teams_raw.values():
            if isinstance(team_info, dict):
                keepers_raw = team_info.get("keepers", [])
                if isinstance(keepers_raw, (list, tuple)):
                    keeper_ids.update(str(k) for k in keepers_raw)
    return keeper_ids


def _build_candidates(
    candidate_ids: list[str],
    all_player_values: list[PlayerValue],
    player_positions: dict[tuple[str, str], tuple[str, ...]],
    yahoo_positions: dict[str, tuple[str, ...]] | None = None,
) -> list[KeeperCandidate]:
    """Build KeeperCandidate list from IDs, matching against player values and positions."""
    pv_by_id: dict[str, PlayerValue] = {}
    for pv in all_player_values:
        # Keep the highest-value entry per player_id (a player may appear as both B and P)
        if pv.player_id not in pv_by_id or pv.total_value > pv_by_id[pv.player_id].total_value:
            pv_by_id[pv.player_id] = pv

    candidates: list[KeeperCandidate] = []
    for cid in candidate_ids:
        pv = pv_by_id.get(cid)
        if pv is None:
            typer.echo(f"Unknown candidate ID: {cid}", err=True)
            raise typer.Exit(code=1)

        if yahoo_positions is not None and cid in yahoo_positions:
            eligible = yahoo_positions[cid]
        else:
            # Collect positions from composite keys
            positions: list[str] = []
            for (pid, _), pos in player_positions.items():
                if pid == cid:
                    positions.extend(pos)
            eligible = tuple(dict.fromkeys(positions))  # deduplicate, preserve order

        candidates.append(
            KeeperCandidate(
                player_id=cid,
                name=pv.name,
                player_value=pv,
                eligible_positions=eligible,
            )
        )

    return candidates


def _resolve_yahoo_inputs(
    no_cache: bool,
    league_id: str | None,
    season: int | None,
    candidates_str: str,
    keepers_file: Path | None,
    teams: int,
) -> tuple[list[str], set[str], int, dict[str, tuple[str, ...]]]:
    """Fetch keeper data from Yahoo rosters.

    Returns (candidate_ids, other_keepers, teams, yahoo_positions).
    """
    apply_cli_overrides(league_id, season)
    try:
        roster_source, league = _get_roster_source_and_league(no_cache=no_cache)
        id_mapper = _get_id_mapper(no_cache=no_cache)
        user_team_key: str = league.team_key()  # type: ignore[union-attr]

        yahoo_source = YahooKeeperSource(
            roster_source=roster_source,
            id_mapper=id_mapper,
            user_team_key=user_team_key,
        )
        yahoo_data = yahoo_source.fetch_keeper_data()

        if yahoo_data.unmapped_yahoo_ids:
            typer.echo(
                f"Warning: {len(yahoo_data.unmapped_yahoo_ids)} player(s) could not be mapped "
                f"to FanGraphs IDs: {', '.join(yahoo_data.unmapped_yahoo_ids)}",
                err=True,
            )

        candidate_ids = list(yahoo_data.user_candidate_ids)

        # If --candidates also provided, use as filter (intersection)
        if candidates_str.strip():
            filter_ids = {c.strip() for c in candidates_str.split(",") if c.strip()}
            candidate_ids = [cid for cid in candidate_ids if cid in filter_ids]

        # If --keepers provided, use YAML file; otherwise use Yahoo other-keepers
        other_keepers = _load_keepers_file(keepers_file) if keepers_file else set(yahoo_data.other_keeper_ids)

        # Derive team count from roster count if not explicitly set
        rosters = roster_source.fetch_rosters()
        num_teams = len(rosters.teams) if teams == 12 else teams

        return candidate_ids, other_keepers, num_teams, dict(yahoo_data.user_candidate_positions)
    finally:
        clear_cli_overrides()


@keeper_app.command(name="rank")
def keeper_rank(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    candidates: Annotated[str, typer.Option("--candidates", help="Comma-separated player IDs.")] = "",
    keepers_file: Annotated[
        Path | None, typer.Option("--keepers", help="YAML file with other teams' keepers.")
    ] = None,
    user_pick: Annotated[int, typer.Option("--user-pick", help="User's draft position (1-based).")] = 5,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    keeper_slots: Annotated[int, typer.Option("--keeper-slots", help="Number of keeper slots.")] = 4,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    yahoo: Annotated[bool, typer.Option("--yahoo", help="Fetch candidates from Yahoo roster.")] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data.")
    ] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Rank keeper candidates by surplus value over draft replacement level."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    yahoo_positions: dict[str, tuple[str, ...]] | None = None

    if yahoo:
        candidate_ids, other_keepers, teams, yahoo_positions = _resolve_yahoo_inputs(
            no_cache, league_id, season, candidates, keepers_file, teams,
        )
    else:
        if not candidates.strip():
            typer.echo("No candidate IDs provided. Use --candidates 'id1,id2,...' or --yahoo", err=True)
            raise typer.Exit(code=1)
        candidate_ids = [c.strip() for c in candidates.split(",") if c.strip()]
        other_keepers = set()
        if keepers_file:
            other_keepers = _load_keepers_file(keepers_file)

    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)

    candidate_list = _build_candidates(candidate_ids, all_values, composite_positions, yahoo_positions)

    calc = DraftPoolReplacementCalculator(user_pick_position=user_pick)
    surplus_calc = SurplusCalculator(calc, num_teams=teams, num_keeper_slots=keeper_slots)
    ranked = surplus_calc.rank_candidates(candidate_list, all_values, other_keepers)

    _display_table(ranked, year, "Keeper Candidates Ranked by Surplus Value")


@keeper_app.command(name="optimize")
def keeper_optimize(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    candidates: Annotated[str, typer.Option("--candidates", help="Comma-separated player IDs.")] = "",
    keepers_file: Annotated[
        Path | None, typer.Option("--keepers", help="YAML file with other teams' keepers.")
    ] = None,
    user_pick: Annotated[int, typer.Option("--user-pick", help="User's draft position (1-based).")] = 5,
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in the league.")] = 12,
    keeper_slots: Annotated[int, typer.Option("--keeper-slots", help="Number of keeper slots.")] = 4,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    yahoo: Annotated[bool, typer.Option("--yahoo", help="Fetch candidates from Yahoo roster.")] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data.")
    ] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Find the optimal keeper combination that maximizes total surplus."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    yahoo_positions: dict[str, tuple[str, ...]] | None = None

    if yahoo:
        candidate_ids, other_keepers, teams, yahoo_positions = _resolve_yahoo_inputs(
            no_cache, league_id, season, candidates, keepers_file, teams,
        )
    else:
        if not candidates.strip():
            typer.echo("No candidate IDs provided. Use --candidates 'id1,id2,...' or --yahoo", err=True)
            raise typer.Exit(code=1)
        candidate_ids = [c.strip() for c in candidates.split(",") if c.strip()]
        other_keepers = set()
        if keepers_file:
            other_keepers = _load_keepers_file(keepers_file)

    typer.echo(f"Generating projections for {year} using {engine}...")
    all_values, composite_positions = build_projections_and_positions(engine, year)

    candidate_list = _build_candidates(candidate_ids, all_values, composite_positions, yahoo_positions)

    calc = DraftPoolReplacementCalculator(user_pick_position=user_pick)
    surplus_calc = SurplusCalculator(calc, num_teams=teams, num_keeper_slots=keeper_slots)
    result = surplus_calc.find_optimal_keepers(candidate_list, all_values, other_keepers)

    # Display recommended keepers
    lines: list[str] = []
    lines.append(f"\nOptimal Keepers for {year} (Total Surplus: {result.total_surplus:.1f}):")
    lines.append("")
    _append_table_rows(lines, list(result.keepers))

    # Display all candidates
    lines.append("")
    lines.append("All Candidates:")
    lines.append("")
    _append_table_rows(lines, list(result.all_candidates))

    typer.echo("\n".join(lines))


def _display_table(ranked: list, year: int, title: str) -> None:
    lines: list[str] = []
    lines.append(f"\n{title} ({year}):")
    lines.append("")
    _append_table_rows(lines, ranked)
    typer.echo("\n".join(lines))


def _append_table_rows(lines: list[str], rows: list) -> None:
    header = f"{'Rk':>4} {'Name':<25} {'Pos':<12} {'Value':>7} {'Repl':>7} {'Surplus':>8} {'Slot':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, ks in enumerate(rows, start=1):
        pos_str = "/".join(ks.eligible_positions) if ks.eligible_positions else "-"
        lines.append(
            f"{i:>4} {ks.name:<25} {pos_str:<12}"
            f" {ks.player_value:>7.1f} {ks.replacement_value:>7.1f}"
            f" {ks.surplus_value:>8.1f} {ks.assigned_slot:>5}"
        )
