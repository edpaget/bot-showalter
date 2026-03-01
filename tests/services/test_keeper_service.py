from fantasy_baseball_manager.domain import Player
from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.services.keeper_service import set_keeper_cost
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
