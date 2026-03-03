from fantasy_baseball_manager.domain import KeeperCost, Player
from fantasy_baseball_manager.services.keeper_history import build_keeper_histories


def _make_cost(
    player_id: int, season: int, *, league: str = "keeper", cost: float = 10.0, source: str = "yahoo_draft"
) -> KeeperCost:
    return KeeperCost(player_id=player_id, season=season, league=league, cost=cost, source=source)


def _make_player(player_id: int, first: str = "Test", last: str = "Player") -> Player:
    return Player(id=player_id, name_first=first, name_last=last)


class TestBuildKeeperHistories:
    def test_groups_by_player(self) -> None:
        costs = [
            _make_cost(100, 2024),
            _make_cost(100, 2025),
            _make_cost(200, 2025),
        ]
        players = [_make_player(100, "Mike", "Trout"), _make_player(200, "Shohei", "Ohtani")]

        result = build_keeper_histories(costs, players, "keeper")

        assert len(result) == 2
        histories_by_id = {h.player_id: h for h in result}
        assert len(histories_by_id[100].entries) == 2
        assert len(histories_by_id[200].entries) == 1

    def test_filters_by_league(self) -> None:
        costs = [
            _make_cost(100, 2025, league="keeper"),
            _make_cost(100, 2025, league="redraft"),
        ]
        players = [_make_player(100)]

        result = build_keeper_histories(costs, players, "keeper")

        assert len(result) == 1
        assert result[0].league == "keeper"
        assert len(result[0].entries) == 1

    def test_sorts_by_season(self) -> None:
        costs = [
            _make_cost(100, 2026),
            _make_cost(100, 2024),
            _make_cost(100, 2025),
        ]
        players = [_make_player(100)]

        result = build_keeper_histories(costs, players, "keeper")

        seasons = [e.season for e in result[0].entries]
        assert seasons == [2024, 2025, 2026]

    def test_player_name_populated(self) -> None:
        costs = [_make_cost(100, 2025)]
        players = [_make_player(100, "Mike", "Trout")]

        result = build_keeper_histories(costs, players, "keeper")

        assert result[0].player_name == "Mike Trout"

    def test_empty_returns_empty(self) -> None:
        result = build_keeper_histories([], [], "keeper")

        assert result == []

    def test_single_season(self) -> None:
        costs = [_make_cost(100, 2025, source="yahoo_trade")]
        players = [_make_player(100, "Aaron", "Judge")]

        result = build_keeper_histories(costs, players, "keeper")

        assert len(result) == 1
        assert result[0].entries[0].season == 2025
        assert result[0].entries[0].source == "yahoo_trade"

    def test_unknown_player_name(self) -> None:
        costs = [_make_cost(999, 2025)]
        players: list[Player] = []

        result = build_keeper_histories(costs, players, "keeper")

        assert len(result) == 1
        assert result[0].player_name == "Unknown"
