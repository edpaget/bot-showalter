"""Helper functions for keeper CLI commands.

This module contains shared logic for loading keepers, resolving Yahoo inputs,
and building keeper candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import typer
import yaml

from fantasy_baseball_manager.keeper.models import KeeperCandidate
from fantasy_baseball_manager.keeper.yahoo_source import LeagueKeeperData, YahooKeeperSource
from fantasy_baseball_manager.services import cli_context, get_container

if TYPE_CHECKING:
    from pathlib import Path

    import yahoo_fantasy_api

    from fantasy_baseball_manager.valuation.models import PlayerValue


def load_keepers_file(path: Path) -> set[str]:
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


def build_candidates(
    candidate_ids: list[str],
    all_player_values: list[PlayerValue],
    player_positions: dict[tuple[str, str], tuple[str, ...]],
    yahoo_positions: dict[tuple[str, str], tuple[str, ...]] | None = None,
    *,
    candidate_position_types: list[str] | None = None,
    strict: bool = True,
) -> list[KeeperCandidate]:
    """Build KeeperCandidate list from IDs, matching against player values and positions.

    When *strict* is True (default), unknown IDs cause an error exit.
    When False, unknown IDs are silently skipped (useful for league-wide processing
    where some rostered players may lack projections).
    """
    pv_by_id: dict[str, PlayerValue] = {}
    for pv in all_player_values:
        # Keep the highest-value entry per player_id (a player may appear as both B and P)
        if pv.player_id not in pv_by_id or pv.total_value > pv_by_id[pv.player_id].total_value:
            pv_by_id[pv.player_id] = pv

    # Build a lookup keyed by (player_id, position_type) for split-player resolution
    pv_by_key: dict[tuple[str, str], PlayerValue] = {}
    for pv in all_player_values:
        key = (pv.player_id, pv.position_type)
        if key not in pv_by_key or pv.total_value > pv_by_key[key].total_value:
            pv_by_key[key] = pv

    candidates: list[KeeperCandidate] = []
    for i, cid in enumerate(candidate_ids):
        pos_type = candidate_position_types[i] if candidate_position_types is not None else None

        # Resolve the correct PlayerValue: prefer (id, position_type) match, fall back to best
        pv: PlayerValue | None = None
        if pos_type is not None:
            pv = pv_by_key.get((cid, pos_type))
        if pv is None:
            pv = pv_by_id.get(cid)
        if pv is None:
            if strict:
                typer.echo(f"Unknown candidate ID: {cid}", err=True)
                raise typer.Exit(code=1)
            continue

        if yahoo_positions is not None and pos_type is not None and (cid, pos_type) in yahoo_positions:
            eligible = yahoo_positions[(cid, pos_type)]
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


def resolve_yahoo_inputs(
    no_cache: bool,
    league_id: str | None,
    season: int | None,
    candidates_str: str,
    keepers_file: Path | None,
    teams: int,
) -> tuple[list[str], set[str], int, dict[tuple[str, str], tuple[str, ...]], list[str]]:
    """Fetch keeper data from Yahoo rosters.

    Returns (candidate_ids, other_keepers, teams, yahoo_positions, candidate_position_types).
    """
    with cli_context(league_id=league_id, season=season, no_cache=no_cache):
        container = get_container()
        league = cast("yahoo_fantasy_api.League", container.roster_league)
        user_team_key: str = league.team_key()

        yahoo_source = YahooKeeperSource(
            roster_source=container.roster_source,
            id_mapper=container.id_mapper,
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
        position_types = list(yahoo_data.user_candidate_position_types)

        # If --candidates also provided, use as filter (intersection)
        if candidates_str.strip():
            filter_ids = {c.strip() for c in candidates_str.split(",") if c.strip()}
            filtered_pairs = [
                (cid, pt) for cid, pt in zip(candidate_ids, position_types, strict=True) if cid in filter_ids
            ]
            candidate_ids = [cid for cid, _ in filtered_pairs]
            position_types = [pt for _, pt in filtered_pairs]

        # If --keepers provided, use YAML file; otherwise use Yahoo other-keepers
        other_keepers = load_keepers_file(keepers_file) if keepers_file else set(yahoo_data.other_keeper_ids)

        # Derive team count from roster count if not explicitly set
        rosters = container.roster_source.fetch_rosters()
        num_teams = len(rosters.teams) if teams == 12 else teams

        return candidate_ids, other_keepers, num_teams, dict(yahoo_data.user_candidate_positions), position_types


def resolve_league_inputs(
    no_cache: bool,
    league_id: str | None,
    season: int | None,
    teams: int,
) -> tuple[LeagueKeeperData, int]:
    """Fetch league-wide keeper data from Yahoo rosters.

    Returns (league_keeper_data, num_teams).
    """
    with cli_context(league_id=league_id, season=season, no_cache=no_cache):
        container = get_container()

        yahoo_source = YahooKeeperSource(
            roster_source=container.roster_source,
            id_mapper=container.id_mapper,
            user_team_key="",  # not used by fetch_league_keeper_data
        )
        league_data = yahoo_source.fetch_league_keeper_data()

        if league_data.unmapped_yahoo_ids:
            typer.echo(
                f"Warning: {len(league_data.unmapped_yahoo_ids)} player(s) could not be mapped "
                f"to FanGraphs IDs: {', '.join(league_data.unmapped_yahoo_ids)}",
                err=True,
            )

        rosters = container.roster_source.fetch_rosters()
        num_teams = len(rosters.teams) if teams == 12 else teams

        return league_data, num_teams
