import numpy as np
import pytest

from fantasy_baseball_manager.domain.projection import StatDistribution
from fantasy_baseball_manager.models.distributions import (
    samples_to_distribution,
    samples_to_distributions,
)


class TestSamplesToDistribution:
    def test_returns_stat_distribution(self) -> None:
        result = samples_to_distribution("hr", [10.0, 20.0, 30.0])
        assert isinstance(result, StatDistribution)
        assert result.stat == "hr"

    def test_percentiles_from_known_data(self) -> None:
        samples = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        result = samples_to_distribution("hr", samples)
        expected = np.percentile(samples, [10, 25, 50, 75, 90])
        assert result.p10 == pytest.approx(expected[0])
        assert result.p25 == pytest.approx(expected[1])
        assert result.p50 == pytest.approx(expected[2])
        assert result.p75 == pytest.approx(expected[3])
        assert result.p90 == pytest.approx(expected[4])

    def test_mean_and_std(self) -> None:
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = samples_to_distribution("rbi", samples)
        assert result.mean == pytest.approx(float(np.mean(samples)))
        assert result.std == pytest.approx(float(np.std(samples)))

    def test_small_sample(self) -> None:
        result = samples_to_distribution("hr", [5.0, 15.0])
        assert isinstance(result, StatDistribution)
        assert result.p10 == pytest.approx(float(np.percentile([5.0, 15.0], 10)))
        assert result.p90 == pytest.approx(float(np.percentile([5.0, 15.0], 90)))

    def test_single_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            samples_to_distribution("hr", [42.0])

    def test_empty_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            samples_to_distribution("hr", [])

    def test_skewed_data(self) -> None:
        # Heavy right skew: many small values, few large
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
        result = samples_to_distribution("hr", samples)
        # Median should be less than mean for right-skewed data
        assert result.p50 < result.mean  # type: ignore[operator]
        # Right tail (p90-p50) should be wider than left tail (p50-p10)
        assert (result.p90 - result.p50) > (result.p50 - result.p10)

    def test_family_is_none(self) -> None:
        result = samples_to_distribution("hr", [10.0, 20.0, 30.0])
        assert result.family is None


class TestSamplesToDistributions:
    def test_batch_single_stat(self) -> None:
        result = samples_to_distributions({"hr": [10.0, 20.0, 30.0]})
        assert len(result) == 1
        assert "hr" in result
        assert isinstance(result["hr"], StatDistribution)
        assert result["hr"].stat == "hr"

    def test_batch_multiple_stats(self) -> None:
        result = samples_to_distributions(
            {
                "hr": [10.0, 20.0, 30.0],
                "rbi": [50.0, 60.0, 70.0],
            }
        )
        assert len(result) == 2
        assert result["hr"].stat == "hr"
        assert result["rbi"].stat == "rbi"

    def test_batch_empty_dict(self) -> None:
        result = samples_to_distributions({})
        assert result == {}
