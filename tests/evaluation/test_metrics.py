import pytest

from fantasy_baseball_manager.evaluation.metrics import (
    mae,
    pearson_r,
    rmse,
    spearman_rho,
    top_n_precision,
)


class TestRmse:
    def test_perfect_prediction(self) -> None:
        assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self) -> None:
        # errors: 1, -1 -> squared: 1, 1 -> mean: 1 -> sqrt: 1
        assert rmse([2.0, 3.0], [1.0, 4.0]) == pytest.approx(1.0)

    def test_single_element(self) -> None:
        assert rmse([5.0], [3.0]) == pytest.approx(2.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            rmse([], [])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            rmse([1.0], [1.0, 2.0])


class TestMae:
    def test_perfect_prediction(self) -> None:
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self) -> None:
        # errors: |1|, |-1| -> mean: 1
        assert mae([2.0, 3.0], [1.0, 4.0]) == pytest.approx(1.0)

    def test_single_element(self) -> None:
        assert mae([5.0], [3.0]) == pytest.approx(2.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mae([], [])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            mae([1.0], [1.0, 2.0])


class TestPearsonR:
    def test_perfect_positive(self) -> None:
        assert pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        assert pearson_r([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)

    def test_zero_std_returns_zero(self) -> None:
        assert pearson_r([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self) -> None:
        # x=[1,2,3], y=[1,3,2]: r = 0.5
        assert pearson_r([1.0, 2.0, 3.0], [1.0, 3.0, 2.0]) == pytest.approx(0.5)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            pearson_r([], [])

    def test_single_element_returns_zero(self) -> None:
        # std is zero for a single element
        assert pearson_r([1.0], [2.0]) == 0.0


class TestSpearmanRho:
    def test_perfect_agreement(self) -> None:
        assert spearman_rho([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        assert spearman_rho([10.0, 20.0, 30.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_with_ties(self) -> None:
        # With ties, average ranks are used
        # x=[10, 10, 30] -> ranks [1.5, 1.5, 3]
        # y=[1, 2, 3] -> ranks [1, 2, 3]
        result = spearman_rho([10.0, 10.0, 30.0], [1.0, 2.0, 3.0])
        assert result == pytest.approx(0.866025, abs=1e-4)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            spearman_rho([], [])


class TestTopNPrecision:
    def test_perfect_overlap(self) -> None:
        assert top_n_precision(["a", "b", "c", "d"], ["a", "b", "c", "d"], 3) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        assert top_n_precision(["a", "b", "c"], ["d", "e", "f"], 3) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        # projected top 2: a, b; actual top 2: a, c -> 1 overlap / 2 = 0.5
        assert top_n_precision(["a", "b", "c"], ["a", "c", "b"], 2) == pytest.approx(0.5)

    def test_n_larger_than_lists(self) -> None:
        assert top_n_precision(["a", "b"], ["a", "b"], 5) == pytest.approx(1.0)

    def test_n_zero(self) -> None:
        assert top_n_precision(["a", "b"], ["a", "b"], 0) == pytest.approx(0.0)
