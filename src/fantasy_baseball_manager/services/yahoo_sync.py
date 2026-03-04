from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import YahooLeague
    from fantasy_baseball_manager.repos import (
        YahooLeagueRepo,
        YahooLeagueSourceProto,
        YahooTeamRepo,
        YahooTransactionRepo,
        YahooTransactionSourceProto,
    )


def sync_league_metadata(
    league_source: YahooLeagueSourceProto,
    league_repo: YahooLeagueRepo,
    team_repo: YahooTeamRepo,
    league_key: str,
    game_key: str,
) -> YahooLeague:
    """Fetch and upsert league + team metadata. Returns the league.

    Does NOT commit — caller is responsible for committing.
    """
    league, teams = league_source.fetch(league_key=league_key, game_key=game_key)
    league_repo.upsert(league)
    for team in teams:
        team_repo.upsert(team)
    return league


def sync_transactions(
    transaction_source: YahooTransactionSourceProto,
    transaction_repo: YahooTransactionRepo,
    league_key: str,
) -> int:
    """Incrementally fetch and store new transactions. Returns count.

    Does NOT commit — caller is responsible for committing.
    """
    latest = transaction_repo.get_latest_timestamp(league_key)
    new = transaction_source.fetch_transactions(league_key, since=latest)
    for txn, players in new:
        transaction_repo.upsert(txn, players)
    return len(new)
