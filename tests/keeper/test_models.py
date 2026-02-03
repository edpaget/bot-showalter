import pytest

from fantasy_baseball_manager.keeper.models import (
    KeeperCandidate,
    KeeperRecommendation,
    KeeperSurplus,
    SlotCost,
)
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, StatCategory


class TestSlotCost:
    def test_construction(self) -> None:
        sc = SlotCost(slot_number=1, replacement_value=5.0)
        assert sc.slot_number == 1
        assert sc.replacement_value == 5.0

    def test_frozen(self) -> None:
        sc = SlotCost(slot_number=1, replacement_value=5.0)
        with pytest.raises(AttributeError):
            sc.slot_number = 2  # type: ignore[misc]


class TestKeeperCandidate:
    def test_construction(self) -> None:
        pv = PlayerValue(
            player_id="p1",
            name="Player One",
            category_values=(),
            total_value=10.0,
        )
        kc = KeeperCandidate(
            player_id="p1",
            name="Player One",
            player_value=pv,
            eligible_positions=("SS", "2B"),
        )
        assert kc.player_id == "p1"
        assert kc.eligible_positions == ("SS", "2B")
        assert kc.player_value.total_value == 10.0

    def test_frozen(self) -> None:
        pv = PlayerValue(player_id="p1", name="P", category_values=(), total_value=0.0)
        kc = KeeperCandidate(player_id="p1", name="P", player_value=pv, eligible_positions=())
        with pytest.raises(AttributeError):
            kc.name = "other"  # type: ignore[misc]


class TestKeeperSurplus:
    def test_construction(self) -> None:
        cv = CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=2.5)
        ks = KeeperSurplus(
            player_id="p1",
            name="Slugger",
            player_value=10.0,
            eligible_positions=("1B",),
            assigned_slot=1,
            replacement_value=6.0,
            surplus_value=4.0,
            category_values=(cv,),
        )
        assert ks.surplus_value == 4.0
        assert ks.category_values[0].category == StatCategory.HR

    def test_frozen(self) -> None:
        ks = KeeperSurplus(
            player_id="p1",
            name="P",
            player_value=0.0,
            eligible_positions=(),
            assigned_slot=1,
            replacement_value=0.0,
            surplus_value=0.0,
            category_values=(),
        )
        with pytest.raises(AttributeError):
            ks.surplus_value = 99.0  # type: ignore[misc]


class TestKeeperRecommendation:
    def test_construction(self) -> None:
        ks = KeeperSurplus(
            player_id="p1",
            name="P",
            player_value=10.0,
            eligible_positions=(),
            assigned_slot=1,
            replacement_value=6.0,
            surplus_value=4.0,
            category_values=(),
        )
        rec = KeeperRecommendation(
            keepers=(ks,),
            total_surplus=4.0,
            all_candidates=(ks,),
        )
        assert rec.total_surplus == 4.0
        assert len(rec.keepers) == 1

    def test_frozen(self) -> None:
        rec = KeeperRecommendation(keepers=(), total_surplus=0.0, all_candidates=())
        with pytest.raises(AttributeError):
            rec.total_surplus = 1.0  # type: ignore[misc]
