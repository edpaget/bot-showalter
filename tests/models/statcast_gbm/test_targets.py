from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS


class TestBatterTargets:
    def test_is_tuple(self) -> None:
        assert isinstance(BATTER_TARGETS, tuple)

    def test_not_empty(self) -> None:
        assert len(BATTER_TARGETS) > 0

    def test_all_strings(self) -> None:
        assert all(isinstance(t, str) for t in BATTER_TARGETS)


class TestPitcherTargets:
    def test_is_tuple(self) -> None:
        assert isinstance(PITCHER_TARGETS, tuple)

    def test_not_empty(self) -> None:
        assert len(PITCHER_TARGETS) > 0

    def test_all_strings(self) -> None:
        assert all(isinstance(t, str) for t in PITCHER_TARGETS)
