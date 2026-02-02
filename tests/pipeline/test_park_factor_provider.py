import pytest

from fantasy_baseball_manager.pipeline.park_factors import FanGraphsParkFactorProvider


class TestFanGraphsParkFactorProvider:
    def test_regression_toward_one(self) -> None:
        """With regression_weight=0.5, a raw factor of 1.2 becomes 1.1."""
        provider = FanGraphsParkFactorProvider(regression_weight=0.5)
        result = provider._regress(1.2)
        assert result == pytest.approx(1.1)

    def test_regression_weight_zero_returns_one(self) -> None:
        """With regression_weight=0.0, all factors become 1.0."""
        provider = FanGraphsParkFactorProvider(regression_weight=0.0)
        result = provider._regress(1.5)
        assert result == pytest.approx(1.0)

    def test_regression_weight_one_returns_raw(self) -> None:
        """With regression_weight=1.0, raw factor is returned unchanged."""
        provider = FanGraphsParkFactorProvider(regression_weight=1.0)
        result = provider._regress(1.3)
        assert result == pytest.approx(1.3)

    def test_column_map_has_expected_stats(self) -> None:
        expected = {"hr", "singles", "doubles", "triples", "bb", "so", "r"}
        assert set(FanGraphsParkFactorProvider._column_map().values()) == expected
