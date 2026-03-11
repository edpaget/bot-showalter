from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import KeeperCostRepo, LeagueKeeperRepo


def migrate_league_name(
    keeper_repo: KeeperCostRepo,
    league_keeper_repo: LeagueKeeperRepo,
    from_league: str,
    to_league: str,
) -> tuple[int, int]:
    """Rename league in keeper_cost and league_keeper tables.

    Returns (keeper_cost_count, league_keeper_count) of rows updated.
    """
    kc_count = keeper_repo.rename_league(from_league, to_league)
    lk_count = league_keeper_repo.rename_league(from_league, to_league)
    return kc_count, lk_count
