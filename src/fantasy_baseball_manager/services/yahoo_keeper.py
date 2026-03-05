from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import KeeperCost
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

    user_team = team_repo.get_user_team(prior_league_key)
    if user_team is None:
        msg = f"No user team found for {prior_league_key}. Run 'fbm yahoo sync --season {prior_season}' first."
        raise ValueError(msg)

    today = datetime.date.today()
    roster = roster_source.fetch_team_roster(
        team_key=user_team.team_key,
        league_key=prior_league_key,
        season=prior_season,
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


def derive_best_n_keeper_costs(
    roster_source: YahooRosterSourceProto,
    team_repo: YahooTeamRepo,
    keeper_repo: KeeperCostRepo,
    prior_league_key: str,
    prior_season: int,
    season: int,
    league_name: str,
) -> None:
    """Derive $0 keeper costs for all roster players (best-N format).

    In best-N leagues, all players cost $0 and the optimizer picks the top N.
    Respects manual overrides (non-best_n source entries).
    Does NOT commit — caller is responsible for committing.
    """
    user_team = team_repo.get_user_team(prior_league_key)
    if user_team is None:
        msg = f"No user team found for {prior_league_key}. Run 'fbm yahoo sync --season {prior_season}' first."
        raise ValueError(msg)

    today = datetime.date.today()
    roster = roster_source.fetch_team_roster(
        team_key=user_team.team_key,
        league_key=prior_league_key,
        season=prior_season,
        week=1,
        as_of=today,
    )

    derived: list[KeeperCost] = []
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
