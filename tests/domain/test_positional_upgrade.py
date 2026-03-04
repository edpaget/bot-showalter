from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.positional_upgrade import (
    MarginalValue,
    RosterSlot,
    RosterState,
)


class TestRosterSlot:
    def test_defaults(self) -> None:
        slot = RosterSlot(position="C")
        assert slot.position == "C"
        assert slot.player_id is None
        assert slot.player_name is None
        assert slot.value == 0.0
        assert slot.category_z_scores == {}

    def test_filled_slot(self) -> None:
        slot = RosterSlot(
            position="SS",
            player_id=42,
            player_name="Trea Turner",
            value=25.0,
            category_z_scores={"HR": 0.5, "SB": 2.1},
        )
        assert slot.player_id == 42
        assert slot.value == 25.0
        assert slot.category_z_scores["SB"] == 2.1

    def test_frozen(self) -> None:
        slot = RosterSlot(position="C")
        with pytest.raises(FrozenInstanceError):
            slot.position = "1B"  # type: ignore[misc]


class TestRosterState:
    def test_construction(self) -> None:
        slots = [RosterSlot(position="C"), RosterSlot(position="1B")]
        state = RosterState(
            slots=slots,
            open_positions=["C", "1B"],
            total_value=0.0,
            category_totals={"HR": 0.0},
        )
        assert len(state.slots) == 2
        assert state.open_positions == ["C", "1B"]
        assert state.total_value == 0.0

    def test_frozen(self) -> None:
        state = RosterState(slots=[], open_positions=[], total_value=0.0, category_totals={})
        with pytest.raises(FrozenInstanceError):
            state.total_value = 10.0  # type: ignore[misc]


class TestMarginalValue:
    def test_construction(self) -> None:
        mv = MarginalValue(
            player_id=1,
            player_name="Juan Soto",
            position="OF",
            raw_value=30.0,
            marginal_value=35.0,
            category_impacts={"HR": 1.5, "AVG": 0.8},
            fills_need=True,
        )
        assert mv.player_id == 1
        assert mv.marginal_value == 35.0
        assert mv.fills_need is True
        assert mv.upgrade_over is None

    def test_upgrade_over(self) -> None:
        mv = MarginalValue(
            player_id=2,
            player_name="Corey Seager",
            position="SS",
            raw_value=22.0,
            marginal_value=5.0,
            category_impacts={"HR": 0.3},
            fills_need=False,
            upgrade_over="Willy Adames",
        )
        assert mv.upgrade_over == "Willy Adames"
        assert mv.fills_need is False

    def test_frozen(self) -> None:
        mv = MarginalValue(
            player_id=1,
            player_name="X",
            position="C",
            raw_value=0.0,
            marginal_value=0.0,
            category_impacts={},
            fills_need=False,
        )
        with pytest.raises(FrozenInstanceError):
            mv.marginal_value = 99.0  # type: ignore[misc]
