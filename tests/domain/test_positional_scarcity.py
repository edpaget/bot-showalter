from fantasy_baseball_manager.domain.positional_scarcity import (
    PositionScarcity,
    PositionValueCurve,
)


class TestPositionScarcity:
    def test_position_scarcity_frozen(self) -> None:
        ps = PositionScarcity(
            position="ss",
            tier_1_value=25.0,
            replacement_value=5.0,
            total_surplus=100.0,
            dropoff_slope=-1.5,
            steep_rank=8,
        )
        try:
            ps.position = "c"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_position_scarcity_fields(self) -> None:
        ps = PositionScarcity(
            position="c",
            tier_1_value=18.5,
            replacement_value=2.0,
            total_surplus=80.0,
            dropoff_slope=-2.3,
            steep_rank=None,
        )
        assert ps.position == "c"
        assert ps.tier_1_value == 18.5
        assert ps.replacement_value == 2.0
        assert ps.total_surplus == 80.0
        assert ps.dropoff_slope == -2.3
        assert ps.steep_rank is None


class TestPositionValueCurve:
    def test_frozen(self) -> None:
        curve = PositionValueCurve(
            position="ss",
            values=[(1, "Player A", 30.0)],
            cliff_rank=5,
        )
        try:
            curve.position = "c"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_fields(self) -> None:
        values = [(1, "Alice", 25.0), (2, "Bob", 20.0), (3, "Carol", 15.0)]
        curve = PositionValueCurve(
            position="of",
            values=values,
            cliff_rank=2,
        )
        assert curve.position == "of"
        assert curve.values == values
        assert curve.cliff_rank == 2

    def test_cliff_rank_none(self) -> None:
        curve = PositionValueCurve(
            position="1b",
            values=[(1, "Player", 10.0)],
            cliff_rank=None,
        )
        assert curve.cliff_rank is None
