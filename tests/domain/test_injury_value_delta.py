import pytest

from fantasy_baseball_manager.domain.injury_value_delta import InjuryValueDelta


class TestInjuryValueDelta:
    def test_construction(self) -> None:
        delta = InjuryValueDelta(
            player_name="Mike Trout",
            original_value=35.0,
            adjusted_value=25.0,
            value_delta=-10.0,
            original_rank=5,
            adjusted_rank=15,
            rank_change=-10,
            expected_days_lost=45.0,
        )
        assert delta.player_name == "Mike Trout"
        assert delta.original_value == 35.0
        assert delta.adjusted_value == 25.0
        assert delta.value_delta == -10.0
        assert delta.original_rank == 5
        assert delta.adjusted_rank == 15
        assert delta.rank_change == -10
        assert delta.expected_days_lost == 45.0

    def test_frozen(self) -> None:
        delta = InjuryValueDelta(
            player_name="Mike Trout",
            original_value=35.0,
            adjusted_value=25.0,
            value_delta=-10.0,
            original_rank=5,
            adjusted_rank=15,
            rank_change=-10,
            expected_days_lost=45.0,
        )
        with pytest.raises(AttributeError):
            delta.player_name = "Other"  # type: ignore[misc]
