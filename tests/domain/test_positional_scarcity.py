from fantasy_baseball_manager.domain.positional_scarcity import PositionScarcity


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
