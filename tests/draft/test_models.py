from fantasy_baseball_manager.draft.models import (
    DraftPick,
    DraftRanking,
    RosterConfig,
    RosterSlot,
)
from fantasy_baseball_manager.valuation.models import CategoryValue, StatCategory


class TestRosterSlot:
    def test_construction(self) -> None:
        slot = RosterSlot(position="C", count=2)
        assert slot.position == "C"
        assert slot.count == 2

    def test_frozen(self) -> None:
        slot = RosterSlot(position="C", count=2)
        try:
            slot.position = "1B"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestRosterConfig:
    def test_construction(self) -> None:
        config = RosterConfig(slots=(RosterSlot(position="C", count=2), RosterSlot(position="OF", count=5)))
        assert len(config.slots) == 2
        assert config.slots[0].position == "C"


class TestDraftPick:
    def test_construction(self) -> None:
        pick = DraftPick(player_id="123", name="Test Player", is_user=True, position="OF")
        assert pick.player_id == "123"
        assert pick.is_user is True
        assert pick.position == "OF"

    def test_position_none_for_opponent_pick(self) -> None:
        pick = DraftPick(player_id="456", name="Other", is_user=False, position=None)
        assert pick.position is None


class TestDraftRanking:
    def test_construction(self) -> None:
        cats = (CategoryValue(category=StatCategory.HR, raw_stat=30.0, value=1.5),)
        ranking = DraftRanking(
            rank=1,
            player_id="123",
            name="Test Player",
            eligible_positions=("1B", "OF"),
            best_position="1B",
            position_multiplier=1.0,
            raw_value=1.5,
            weighted_value=1.5,
            adjusted_value=1.5,
            category_values=cats,
        )
        assert ranking.rank == 1
        assert ranking.eligible_positions == ("1B", "OF")
        assert ranking.adjusted_value == 1.5

    def test_empty_positions(self) -> None:
        ranking = DraftRanking(
            rank=1,
            player_id="123",
            name="Test Player",
            eligible_positions=(),
            best_position=None,
            position_multiplier=1.0,
            raw_value=1.0,
            weighted_value=1.0,
            adjusted_value=1.0,
            category_values=(),
        )
        assert ranking.eligible_positions == ()
        assert ranking.best_position is None
