from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.services.draft_board import build_draft_board
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    build_draft_roster_slots,
)
from fantasy_baseball_manager.services.draft_translation import (
    build_player_id_aliases,
    build_team_map,
    ingest_yahoo_pick,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoard, DraftBoardRow, LeagueSettings
    from fantasy_baseball_manager.repos import (
        ADPRepo,
        PlayerRepo,
        ValuationRepo,
        YahooDraftRepo,
        YahooDraftSourceProto,
        YahooLeagueRepo,
        YahooTeamRepo,
    )


@dataclass(frozen=True)
class YahooDraftSetup:
    engine: DraftEngine
    board: DraftBoard
    team_map: dict[str, int]
    source: YahooDraftSourceProto
    replayed_count: int
    id_aliases: dict[int, int]


def build_yahoo_draft_setup(
    team_repo: YahooTeamRepo,
    league_repo: YahooLeagueRepo,
    valuation_repo: ValuationRepo,
    player_repo: PlayerRepo,
    adp_repo: ADPRepo,
    draft_repo: YahooDraftRepo,
    draft_source: YahooDraftSourceProto,
    league_key: str,
    season: int,
    fbm_league: LeagueSettings,
    system: str,
    version: str,
    provider: str,
) -> YahooDraftSetup:
    """Build a Yahoo draft setup with engine, board, and replayed picks.

    Raises ValueError if no teams, user team, or league metadata found.
    Does NOT commit — caller is responsible for committing.
    """
    teams = team_repo.get_by_league_key(league_key)
    if not teams:
        msg = f"No teams found for league key '{league_key}'. Run 'fbm yahoo sync' first."
        raise ValueError(msg)

    team_map = build_team_map(teams)

    user_team = team_repo.get_user_team(league_key)
    if user_team is None:
        msg = "No user team found. Run 'fbm yahoo sync' first."
        raise ValueError(msg)

    valuations = valuation_repo.get_by_season(season, system=system, version=version)

    player_ids = [v.player_id for v in valuations]
    players_list = player_repo.get_by_ids(player_ids)
    player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players_list if p.id is not None}

    adp_list = adp_repo.get_by_season(season, provider=provider)

    board = build_draft_board(valuations, fbm_league, player_names, adp=adp_list if adp_list else None)
    draft_players: list[DraftBoardRow] = board.rows

    yahoo_league = league_repo.get_by_league_key(league_key)
    if yahoo_league is None:
        msg = "League metadata not found. Run 'fbm yahoo sync' first."
        raise ValueError(msg)

    roster_slots = build_draft_roster_slots(fbm_league)
    draft_type = yahoo_league.draft_type
    draft_format = DraftFormat.AUCTION if "auction" in draft_type.lower() else DraftFormat.LIVE
    draft_config = DraftConfig(
        teams=yahoo_league.num_teams,
        roster_slots=roster_slots,
        format=draft_format,
        user_team=user_team.team_id,
        season=season,
        budget=260 if draft_format == DraftFormat.AUCTION else 0,
    )

    engine = DraftEngine()
    engine.start(draft_players, draft_config)

    existing_picks = draft_source.fetch_draft_results(league_key, season)
    board_names = {row.player_id: row.player_name for row in board.rows}
    aliases = build_player_id_aliases(existing_picks, board_names)

    for pick in existing_picks:
        draft_repo.upsert(pick)
        ingest_yahoo_pick(
            engine.pick,
            {pid for pid, _ in engine.state.available_pool},
            pick,
            team_map,
            id_aliases=aliases,
            roster_slots=roster_slots,
            team_rosters=engine.state.team_rosters,
        )

    return YahooDraftSetup(
        engine=engine,
        board=board,
        team_map=team_map,
        source=draft_source,
        replayed_count=len(existing_picks),
        id_aliases=aliases,
    )
