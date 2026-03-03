from fantasy_baseball_manager.domain import KeeperCost, RosterEntry, YahooDraftPick

_SOURCE_MAP = {
    "draft": "yahoo_draft",
    "trade": "yahoo_trade",
    "add": "yahoo_fa",
}


def derive_keeper_costs(
    draft_picks: list[YahooDraftPick],
    roster_entries: list[RosterEntry],
    league: str,
    season: int,
    cost_floor: float = 1.0,
) -> list[KeeperCost]:
    pick_lookup: dict[int, YahooDraftPick] = {}
    for pick in draft_picks:
        if pick.player_id is not None:
            pick_lookup[pick.player_id] = pick

    costs: list[KeeperCost] = []
    for entry in roster_entries:
        if entry.player_id is None:
            continue

        pick = pick_lookup.get(entry.player_id)
        if pick is not None:
            cost = float(pick.cost) if pick.cost is not None else float(pick.round)
            source = _SOURCE_MAP.get(entry.acquisition_type, "yahoo_fa")
        else:
            cost = cost_floor
            source = "yahoo_fa"

        costs.append(
            KeeperCost(
                player_id=entry.player_id,
                season=season,
                league=league,
                cost=cost,
                years_remaining=1,
                source=source,
            )
        )

    return costs
