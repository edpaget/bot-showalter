from fantasy_baseball_manager.models.composite.targets import BATTER_TARGETS, PITCHER_TARGETS


class TestBatterTargets:
    def test_contains_expected_stats(self) -> None:
        assert BATTER_TARGETS == ("avg", "obp", "slg", "woba", "iso", "babip")

    def test_length(self) -> None:
        assert len(BATTER_TARGETS) == 6


class TestPitcherTargets:
    def test_contains_expected_stats(self) -> None:
        assert PITCHER_TARGETS == ("era", "fip", "k_per_9", "bb_per_9", "hr_per_9", "babip", "whip")

    def test_length(self) -> None:
        assert len(PITCHER_TARGETS) == 7
