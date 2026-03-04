from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.services.keeper_cost_derivation import derive_keeper_costs

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import (
        KeeperCostRepo,
        YahooDraftSourceProto,
        YahooRosterSourceProto,
        YahooTeamRepo,
    )


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
    """Derive keeper costs from Yahoo draft/roster and upsert, respecting manual overrides.

    Raises ValueError if no user team is found.
    Does NOT commit — caller is responsible for committing.
    """
    draft_picks = draft_source.fetch_draft_results(prior_league_key, prior_season)

    user_team = team_repo.get_user_team(league_key)
    if user_team is None:
        msg = "No user team found. Run 'fbm yahoo sync' first."
        raise ValueError(msg)

    today = datetime.date.today()
    roster = roster_source.fetch_team_roster(
        team_key=user_team.team_key,
        league_key=league_key,
        season=season,
        week=1,
        as_of=today,
    )

    derived = derive_keeper_costs(draft_picks, list(roster.entries), league_name, season, cost_floor)

    # Respect manual overrides: skip upserting players with non-yahoo_* source
    existing = keeper_repo.find_by_season_league(season, league_name)
    manual_player_ids = {kc.player_id for kc in existing if not kc.source.startswith("yahoo_")}
    to_upsert = [kc for kc in derived if kc.player_id not in manual_player_ids]

    if to_upsert:
        keeper_repo.upsert_batch(to_upsert)
