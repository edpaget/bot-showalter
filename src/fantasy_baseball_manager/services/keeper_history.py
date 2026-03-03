from collections import defaultdict

from fantasy_baseball_manager.domain import KeeperCost, KeeperHistory, KeeperSeasonEntry, Player


def build_keeper_histories(
    costs: list[KeeperCost],
    players: list[Player],
    league: str,
) -> list[KeeperHistory]:
    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    grouped: dict[int, list[KeeperCost]] = defaultdict(list)
    for cost in costs:
        if cost.league == league:
            grouped[cost.player_id].append(cost)

    histories: list[KeeperHistory] = []
    for player_id, player_costs in grouped.items():
        player_costs.sort(key=lambda c: c.season)
        player = player_lookup.get(player_id)
        player_name = f"{player.name_first} {player.name_last}" if player else "Unknown"
        entries = tuple(KeeperSeasonEntry(season=c.season, cost=c.cost, source=c.source) for c in player_costs)
        histories.append(
            KeeperHistory(
                player_id=player_id,
                player_name=player_name,
                league=league,
                entries=entries,
            )
        )

    return histories
