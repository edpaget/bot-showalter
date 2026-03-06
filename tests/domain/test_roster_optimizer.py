from fantasy_baseball_manager.domain.roster_optimizer import (
    BudgetAllocation,
    RoundTarget,
    SnakeDraftPlan,
)


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


class TestRoundTarget:
    def test_construction(self) -> None:
        rt = RoundTarget(
            round=1,
            pick_number=3,
            recommended_position="ss",
            target_tier=1,
            expected_value=42.0,
            alternative_positions=("of", "sp"),
        )
        assert rt.round == 1
        assert rt.pick_number == 3
        assert rt.recommended_position == "ss"
        assert rt.target_tier == 1
        assert rt.expected_value == 42.0
        assert rt.alternative_positions == ("of", "sp")

    def test_no_tier(self) -> None:
        rt = RoundTarget(
            round=2,
            pick_number=22,
            recommended_position="of",
            target_tier=None,
            expected_value=15.0,
            alternative_positions=(),
        )
        assert rt.target_tier is None


class TestSnakeDraftPlan:
    def test_construction(self) -> None:
        plan = SnakeDraftPlan(
            draft_slot=1,
            teams=12,
            rounds=[
                RoundTarget(
                    round=1,
                    pick_number=1,
                    recommended_position="of",
                    target_tier=1,
                    expected_value=50.0,
                    alternative_positions=("ss",),
                ),
            ],
        )
        assert plan.draft_slot == 1
        assert plan.teams == 12
        assert len(plan.rounds) == 1
