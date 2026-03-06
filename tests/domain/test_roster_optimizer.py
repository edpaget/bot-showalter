from fantasy_baseball_manager.domain.roster_optimizer import BudgetAllocation


class TestBudgetAllocation:
    def test_frozen_dataclass(self) -> None:
        alloc = BudgetAllocation(
            position="c",
            budget=15.0,
            target_tier=1,
            target_player_names=("J.T. Realmuto",),
        )
        assert alloc.position == "c"
        assert alloc.budget == 15.0
        assert alloc.target_tier == 1
        assert alloc.target_player_names == ("J.T. Realmuto",)

    def test_no_tier(self) -> None:
        alloc = BudgetAllocation(
            position="sp",
            budget=1.0,
            target_tier=None,
            target_player_names=(),
        )
        assert alloc.target_tier is None
