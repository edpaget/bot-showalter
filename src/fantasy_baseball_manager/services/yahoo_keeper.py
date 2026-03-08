from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import KeeperCost
from fantasy_baseball_manager.services.keeper_cost_derivation import derive_keeper_costs
from fantasy_baseball_manager.services.yahoo_sync import sync_league_metadata

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Roster
    from fantasy_baseball_manager.repos import (
        KeeperCostRepo,
        YahooDraftSourceProto,
        YahooLeagueRepo,
        YahooLeagueSourceProto,
        YahooRosterSourceProto,
        YahooTeamRepo,
    )


def ensure_prior_season_teams(
    team_repo: YahooTeamRepo,
    league_source: YahooLeagueSourceProto,
    league_repo: YahooLeagueRepo,
    prior_league_key: str,
    prior_game_key: str,
) -> None:
    """Ensure prior-season teams exist in the DB, syncing from Yahoo if needed.

    Does NOT commit — caller is responsible for committing.
    Raises ValueError if the prior season has no user team after syncing.
    """
    if team_repo.get_user_team(prior_league_key) is not None:
        return

    sync_league_metadata(
        league_source=league_source,
        league_repo=league_repo,
        team_repo=team_repo,
        league_key=prior_league_key,
        game_key=prior_game_key,
    )

    if team_repo.get_user_team(prior_league_key) is None:
        msg = f"No user team found for {prior_league_key}. The prior season may not have this user as a league member."
        raise ValueError(msg)


def derive_and_store_keeper_costs(
    draft_source: YahooDraftSourceProto,
    roster_source: YahooRosterSourceProto,
    team_repo: YahooTeamRepo,
    keeper_repo: KeeperCostRepo,
    league_key: str,
    prior_league_key: str,
    prior_season: int,
    season: int,
    league_name: str,
    cost_floor: float,
) -> None:
    """Derive keeper costs from Yahoo draft/roster for all teams and upsert.

    Respects manual overrides (non-yahoo_* source entries).
    Raises ValueError if no teams are found.
    Does NOT commit — caller is responsible for committing.
    """
    teams = team_repo.get_by_league_key(prior_league_key)
    if not teams:
        msg = f"No teams found for {prior_league_key}."
        raise ValueError(msg)

    draft_picks = draft_source.fetch_draft_results(prior_league_key, prior_season)

    today = datetime.date.today()
    derived: list[KeeperCost] = []
    for team in teams:
        roster = roster_source.fetch_team_roster(
            team_key=team.team_key,
            league_key=prior_league_key,
            season=prior_season,
            as_of=today,
        )
        derived.extend(derive_keeper_costs(draft_picks, list(roster.entries), league_name, season, cost_floor))

    # Respect manual overrides: skip upserting players with non-yahoo_* source
    existing = keeper_repo.find_by_season_league(season, league_name)
    manual_player_ids = {kc.player_id for kc in existing if not kc.source.startswith("yahoo_")}
    to_upsert = [kc for kc in derived if kc.player_id not in manual_player_ids]

    if to_upsert:
        keeper_repo.upsert_batch(to_upsert)


def derive_best_n_keeper_costs(
    roster_source: YahooRosterSourceProto,
    team_repo: YahooTeamRepo,
    keeper_repo: KeeperCostRepo,
    prior_league_key: str,
    prior_season: int,
    season: int,
    league_name: str,
) -> None:
    """Derive $0 keeper costs for all roster players on all teams (best-N format).

    In best-N leagues, all players cost $0 and the optimizer picks the top N.
    Respects manual overrides (non-best_n source entries).
    Does NOT commit — caller is responsible for committing.
    """
    teams = team_repo.get_by_league_key(prior_league_key)
    if not teams:
        msg = f"No teams found for {prior_league_key}."
        raise ValueError(msg)

    today = datetime.date.today()
    derived: list[KeeperCost] = []
    for team in teams:
        roster = roster_source.fetch_team_roster(
            team_key=team.team_key,
            league_key=prior_league_key,
            season=prior_season,
            as_of=today,
        )
        for entry in roster.entries:
            if entry.player_id is None:
                continue
            derived.append(
                KeeperCost(
                    player_id=entry.player_id,
                    season=season,
                    league=league_name,
                    cost=0.0,
                    source="best_n",
                    years_remaining=1,
                )
            )

    # Respect manual overrides
    existing = keeper_repo.find_by_season_league(season, league_name)
    manual_player_ids = {kc.player_id for kc in existing if kc.source != "best_n"}
    to_upsert = [kc for kc in derived if kc.player_id not in manual_player_ids]

    if to_upsert:
        keeper_repo.upsert_batch(to_upsert)


def fetch_league_rosters(
    roster_source: YahooRosterSourceProto,
    team_repo: YahooTeamRepo,
    prior_league_key: str,
    prior_season: int,
) -> list[Roster]:
    """Fetch rosters for all non-user teams in the prior season's league.

    Returns a list of Roster objects, one per non-user team.
    """
    teams = team_repo.get_by_league_key(prior_league_key)
    other_teams = [t for t in teams if not t.is_owned_by_user]

    today = datetime.date.today()
    rosters: list[Roster] = []
    for team in other_teams:
        roster = roster_source.fetch_team_roster(
            team_key=team.team_key,
            league_key=prior_league_key,
            season=prior_season,
            as_of=today,
        )
        rosters.append(roster)

    return rosters
