import pytest

from fantasy_baseball_manager.domain import KeeperCost, Player, Valuation
from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.services.keeper_service import compute_surplus, set_keeper_cost
from tests.fakes.repos import FakeKeeperCostRepo, FakePlayerRepo


class TestSetKeeperCost:
    def test_single_match_returns_ok(self) -> None:
        player = Player(name_first="Mike", name_last="Trout", id=1)
        player_repo = FakePlayerRepo([player])
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Trout", cost=25.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert isinstance(result, Ok)
        kc = result.value
        assert kc.player_id == 1
        assert kc.cost == 25.0
        assert kc.season == 2026
        assert kc.league == "dynasty"
        assert kc.source == "auction"
        assert kc.years_remaining == 1

    def test_single_match_upserts_to_repo(self) -> None:
        player = Player(name_first="Mike", name_last="Trout", id=1)
        player_repo = FakePlayerRepo([player])
        keeper_repo = FakeKeeperCostRepo()

        set_keeper_cost(
            "Trout", cost=25.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        stored = keeper_repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 1
        assert stored[0].player_id == 1

    def test_custom_years_and_source(self) -> None:
        player = Player(name_first="Mike", name_last="Trout", id=1)
        player_repo = FakePlayerRepo([player])
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Trout",
            cost=30.0,
            season=2026,
            league="dynasty",
            player_repo=player_repo,
            keeper_repo=keeper_repo,
            years_remaining=3,
            source="trade",
        )

        assert isinstance(result, Ok)
        assert result.value.years_remaining == 3
        assert result.value.source == "trade"

    def test_no_match_returns_err(self) -> None:
        player_repo = FakePlayerRepo()
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Nobody", cost=10.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert isinstance(result, Err)
        assert "no player found matching 'Nobody'" in result.error

    def test_no_match_does_not_upsert(self) -> None:
        player_repo = FakePlayerRepo()
        keeper_repo = FakeKeeperCostRepo()

        set_keeper_cost(
            "Nobody", cost=10.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert keeper_repo.find_by_season_league(2026, "dynasty") == []

    def test_ambiguous_match_returns_err(self) -> None:
        players = [
            Player(name_first="Mike", name_last="Smith", id=1),
            Player(name_first="John", name_last="Smith", id=2),
        ]
        player_repo = FakePlayerRepo(players)
        keeper_repo = FakeKeeperCostRepo()

        result = set_keeper_cost(
            "Smith", cost=10.0, season=2026, league="dynasty", player_repo=player_repo, keeper_repo=keeper_repo
        )

        assert isinstance(result, Err)
        assert "ambiguous name 'Smith'" in result.error
        assert "Mike Smith" in result.error
        assert "John Smith" in result.error


def _valuation(player_id: int, value: float, position: str = "SS") -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="zar",
        version="v1",
        projection_system="composite",
        projection_version="v1",
        player_type="batter",
        position=position,
        value=value,
        rank=1,
        category_scores={},
    )


def _keeper_cost(player_id: int, cost: float, years: int = 1) -> KeeperCost:
    return KeeperCost(
        player_id=player_id, season=2026, league="dynasty", cost=cost, source="auction", years_remaining=years
    )


class TestComputeSurplus:
    def test_positive_surplus(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].surplus == 15.0
        assert result[0].recommendation == "keep"

    def test_negative_surplus(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 30.0)]
        valuations = [_valuation(1, 10.0)]

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].surplus == -20.0
        assert result[0].recommendation == "release"

    def test_threshold_filters(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 18.0)]
        valuations = [_valuation(1, 20.0)]

        result = compute_surplus(keepers, valuations, players, threshold=5.0)

        assert len(result) == 1
        assert result[0].surplus == 2.0
        assert result[0].recommendation == "release"

    def test_multi_year_discount(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 15.0, years=3)]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players, decay=0.85)

        # single-year surplus = 25 - 15 = 10
        # total = 10 + 10*0.85 + 10*0.85^2 = 10 + 8.5 + 7.225 = 25.725
        assert len(result) == 1
        assert result[0].surplus == pytest.approx(25.725)
        assert result[0].recommendation == "keep"

    def test_sorted_by_surplus_descending(self) -> None:
        players = [
            Player(name_first="Mike", name_last="Trout", id=1),
            Player(name_first="Shohei", name_last="Ohtani", id=2),
            Player(name_first="Aaron", name_last="Judge", id=3),
        ]
        keepers = [_keeper_cost(1, 10.0), _keeper_cost(2, 5.0), _keeper_cost(3, 20.0)]
        valuations = [_valuation(1, 25.0), _valuation(2, 30.0), _valuation(3, 22.0)]

        result = compute_surplus(keepers, valuations, players)

        surpluses = [d.surplus for d in result]
        assert surpluses == sorted(surpluses, reverse=True)
        assert surpluses == [25.0, 15.0, 2.0]

    def test_missing_valuation(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations: list[Valuation] = []

        result = compute_surplus(keepers, valuations, players)

        assert len(result) == 1
        assert result[0].projected_value == 0.0
        assert result[0].surplus == -10.0
        assert result[0].recommendation == "release"

    def test_empty_inputs(self) -> None:
        result = compute_surplus([], [], [])
        assert result == []

    def test_player_name_and_position(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 25.0, position="CF")]

        result = compute_surplus(keepers, valuations, players)

        assert result[0].player_name == "Mike Trout"
        assert result[0].position == "CF"

    def test_missing_player_uses_unknown(self) -> None:
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, [])

        assert result[0].player_name == "Unknown"
        assert result[0].position == "SS"

    def test_highest_valuation_used(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        keepers = [_keeper_cost(1, 10.0)]
        valuations = [_valuation(1, 20.0), _valuation(1, 25.0)]

        result = compute_surplus(keepers, valuations, players)

        assert result[0].projected_value == 25.0
        assert result[0].surplus == 15.0
