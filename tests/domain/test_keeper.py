import pytest

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.keeper import (
    KeeperCost,
    LeagueKeeperOverview,
    ProjectedKeeper,
    TeamKeeperProjection,
    TradeTarget,
)


class TestKeeperCost:
    def test_create_with_required_fields(self) -> None:
        cost = KeeperCost(player_id=1, season=2026, league="dynasty", cost=25.0, source="auction")
        assert cost.player_id == 1
        assert cost.season == 2026
        assert cost.league == "dynasty"
        assert cost.cost == 25.0
        assert cost.source == "auction"

    def test_defaults(self) -> None:
        cost = KeeperCost(player_id=1, season=2026, league="dynasty", cost=10.0, source="auction")
        assert cost.years_remaining == 1
        assert cost.id is None
        assert cost.loaded_at is None

    def test_custom_optional_fields(self) -> None:
        cost = KeeperCost(
            player_id=1,
            season=2026,
            league="dynasty",
            cost=15.0,
            source="contract",
            years_remaining=3,
            id=42,
            loaded_at="2026-02-28T12:00:00",
        )
        assert cost.years_remaining == 3
        assert cost.id == 42
        assert cost.loaded_at == "2026-02-28T12:00:00"

    def test_immutability(self) -> None:
        cost = KeeperCost(player_id=1, season=2026, league="dynasty", cost=25.0, source="auction")
        with pytest.raises(AttributeError):
            cost.cost = 30.0  # type: ignore[misc]


class TestLeagueKeeperDomainModels:
    def test_projected_keeper_construction(self) -> None:
        pk = ProjectedKeeper(
            player_id=1,
            player_name="Mike Trout",
            position="OF",
            player_type=PlayerType.BATTER,
            value=35.0,
            category_scores={"hr": 1.5, "obp": 0.8},
        )
        assert pk.player_id == 1
        assert pk.player_name == "Mike Trout"
        assert pk.category_scores == {"hr": 1.5, "obp": 0.8}

    def test_team_keeper_projection_construction(self) -> None:
        keeper = ProjectedKeeper(
            player_id=1,
            player_name="Mike Trout",
            position="OF",
            player_type=PlayerType.BATTER,
            value=35.0,
            category_scores={"hr": 1.5},
        )
        tkp = TeamKeeperProjection(
            team_key="422.l.1.t.1",
            team_name="Team One",
            is_user=True,
            keepers=(keeper,),
            total_value=35.0,
            category_totals={"hr": 1.5},
        )
        assert tkp.team_name == "Team One"
        assert tkp.is_user is True
        assert len(tkp.keepers) == 1

    def test_trade_target_construction(self) -> None:
        tt = TradeTarget(
            player_id=2,
            player_name="Shohei Ohtani",
            position="DH",
            player_type=PlayerType.BATTER,
            value=40.0,
            owning_team_name="Team Two",
            owning_team_key="422.l.1.t.2",
            rank_on_team=5,
        )
        assert tt.rank_on_team == 5
        assert tt.owning_team_name == "Team Two"

    def test_league_keeper_overview_construction(self) -> None:
        overview = LeagueKeeperOverview(
            team_projections=(),
            trade_targets=(),
            category_names=("hr", "obp"),
        )
        assert overview.category_names == ("hr", "obp")
        assert len(overview.team_projections) == 0
