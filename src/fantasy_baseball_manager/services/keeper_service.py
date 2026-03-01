from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import Err, KeeperCost, KeeperDecision, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Player, Result, Valuation
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


def compute_surplus(
    keeper_costs: list[KeeperCost],
    valuations: list[Valuation],
    players: list[Player],
    threshold: float = 0.0,
    decay: float = 0.85,
) -> list[KeeperDecision]:
    """Compute surplus value for each keeper cost and recommend keep/release.

    Takes pre-fetched data (not repos) — keeps the function pure and easy to test.
    For players with multiple valuations, the highest value is used.
    Multi-year contracts use discounted future surplus: sum(surplus * decay^i).
    """
    # Build lookup: player_id -> highest valuation
    val_lookup: dict[int, Valuation] = {}
    for v in valuations:
        existing = val_lookup.get(v.player_id)
        if existing is None or v.value > existing.value:
            val_lookup[v.player_id] = v

    # Build lookup: player_id -> Player
    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    decisions: list[KeeperDecision] = []
    for kc in keeper_costs:
        val = val_lookup.get(kc.player_id)
        projected_value = val.value if val is not None else 0.0
        position = val.position if val is not None else "UTIL"

        single_year_surplus = projected_value - kc.cost
        total_surplus = sum(single_year_surplus * decay**i for i in range(kc.years_remaining))

        player = player_lookup.get(kc.player_id)
        player_name = f"{player.name_first} {player.name_last}" if player is not None else "Unknown"

        recommendation = "keep" if total_surplus >= threshold else "release"

        decisions.append(
            KeeperDecision(
                player_id=kc.player_id,
                player_name=player_name,
                position=position,
                cost=kc.cost,
                projected_value=projected_value,
                surplus=total_surplus,
                years_remaining=kc.years_remaining,
                recommendation=recommendation,
            )
        )

    decisions.sort(key=lambda d: d.surplus, reverse=True)
    return decisions
