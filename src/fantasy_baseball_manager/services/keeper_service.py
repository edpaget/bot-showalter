from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import Err, KeeperCost, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Result
    from fantasy_baseball_manager.repos import KeeperCostRepo, PlayerRepo


def set_keeper_cost(
    player_name: str,
    cost: float,
    season: int,
    league: str,
    player_repo: PlayerRepo,
    keeper_repo: KeeperCostRepo,
    years_remaining: int = 1,
    source: str = "auction",
) -> Result[KeeperCost, str]:
    """Resolve a player by name and set their keeper cost.

    Returns Ok(KeeperCost) on success, or Err(message) if the player
    cannot be uniquely resolved.
    """
    matches = player_repo.search_by_name(player_name)

    if len(matches) == 0:
        return Err(f"no player found matching '{player_name}'")

    if len(matches) > 1:
        names = [f"{p.name_first} {p.name_last}" for p in matches]
        return Err(f"ambiguous name '{player_name}', matches: {', '.join(names)}")

    player = matches[0]
    assert player.id is not None  # noqa: S101

    keeper_cost = KeeperCost(
        player_id=player.id,
        season=season,
        league=league,
        cost=cost,
        years_remaining=years_remaining,
        source=source,
    )
    keeper_repo.upsert_batch([keeper_cost])
    return Ok(keeper_cost)
