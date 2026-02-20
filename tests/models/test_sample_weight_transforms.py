import math

import pytest

from fantasy_baseball_manager.models.sample_weight_transforms import (
    clamp,
    get_transform,
    log1p,
    raw,
    sqrt,
)


class TestRaw:
    def test_returns_unchanged(self) -> None:
        weights = [1.0, 5.0, 10.0]
        assert raw(weights) == [1.0, 5.0, 10.0]

    def test_empty_list(self) -> None:
        assert raw([]) == []


class TestSqrt:
    def test_sqrt_transform(self) -> None:
        result = sqrt([4.0, 9.0, 16.0])
        assert result == [2.0, 3.0, 4.0]

    def test_handles_one(self) -> None:
        result = sqrt([1.0])
        assert result == [1.0]

    def test_empty_list(self) -> None:
        assert sqrt([]) == []


class TestLog1p:
    def test_log1p_transform(self) -> None:
        result = log1p([0.0, 1.0, 100.0])
        assert math.isclose(result[0], math.log(1), abs_tol=1e-9)
        assert math.isclose(result[1], math.log(2), abs_tol=1e-9)
        assert math.isclose(result[2], math.log(101), abs_tol=1e-9)

    def test_empty_list(self) -> None:
        assert log1p([]) == []


class TestClamp:
    def test_clamps_values(self) -> None:
        transform = clamp(50, 200)
        result = transform([10.0, 100.0, 500.0])
        assert result == [50.0, 100.0, 200.0]

    def test_all_within_range(self) -> None:
        transform = clamp(50, 200)
        result = transform([75.0, 100.0, 150.0])
        assert result == [75.0, 100.0, 150.0]

    def test_empty_list(self) -> None:
        transform = clamp(50, 200)
        assert transform([]) == []


class TestGetTransform:
    def test_known_transforms(self) -> None:
        assert get_transform("raw") is raw
        assert get_transform("sqrt") is sqrt
        assert get_transform("log1p") is log1p

    def test_clamp_variants(self) -> None:
        t = get_transform("clamp_50_200")
        assert t([10.0, 100.0, 500.0]) == [50.0, 100.0, 200.0]

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown sample weight transform"):
            get_transform("bogus")
