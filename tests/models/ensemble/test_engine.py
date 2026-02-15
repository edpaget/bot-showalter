import math

from fantasy_baseball_manager.domain.projection import StatDistribution
from fantasy_baseball_manager.models.ensemble.engine import blend_rates, weighted_average, weighted_spread


class TestWeightedAverage:
    def test_two_systems_equal_weights(self) -> None:
        projections = [
            ({"hr": 30.0, "rbi": 100.0}, 1.0),
            ({"hr": 20.0, "rbi": 80.0}, 1.0),
        ]
        result = weighted_average(projections, stats=["hr", "rbi"])
        assert result == {"hr": 25.0, "rbi": 90.0}

    def test_two_systems_unequal_weights(self) -> None:
        projections = [
            ({"hr": 30.0, "rbi": 100.0}, 0.6),
            ({"hr": 20.0, "rbi": 80.0}, 0.4),
        ]
        result = weighted_average(projections, stats=["hr", "rbi"])
        assert result["hr"] == (30.0 * 0.6 + 20.0 * 0.4) / (0.6 + 0.4)
        assert result["rbi"] == (100.0 * 0.6 + 80.0 * 0.4) / (0.6 + 0.4)

    def test_missing_stat_in_one_system(self) -> None:
        projections = [
            ({"hr": 30.0, "rbi": 100.0}, 0.6),
            ({"hr": 20.0}, 0.4),  # missing rbi
        ]
        result = weighted_average(projections, stats=["hr", "rbi"])
        assert result["hr"] == (30.0 * 0.6 + 20.0 * 0.4) / (0.6 + 0.4)
        # rbi only from one system
        assert result["rbi"] == 100.0

    def test_single_system(self) -> None:
        projections = [
            ({"hr": 25.0, "rbi": 90.0, "sb": 10.0}, 1.0),
        ]
        result = weighted_average(projections, stats=["hr", "rbi", "sb"])
        assert result == {"hr": 25.0, "rbi": 90.0, "sb": 10.0}

    def test_empty_projections(self) -> None:
        result = weighted_average([], stats=["hr", "rbi"])
        assert result == {}

    def test_stat_not_in_any_system(self) -> None:
        projections = [
            ({"hr": 30.0}, 1.0),
            ({"hr": 20.0}, 1.0),
        ]
        result = weighted_average(projections, stats=["hr", "sb"])
        assert result == {"hr": 25.0}


class TestBlendRates:
    def test_two_systems_blend(self) -> None:
        projections = [
            ({"avg": 0.300, "obp": 0.400, "pa": 600.0}, 0.6),
            ({"avg": 0.250, "obp": 0.350, "pa": 500.0}, 0.4),
        ]
        result = blend_rates(projections, rate_stats=["avg", "obp"], pt_stat="pa")
        expected_avg = (0.300 * 0.6 + 0.250 * 0.4) / (0.6 + 0.4)
        expected_obp = (0.400 * 0.6 + 0.350 * 0.4) / (0.6 + 0.4)
        expected_pa = (600.0 * 0.6 + 500.0 * 0.4) / (0.6 + 0.4)
        assert result["avg"] == expected_avg
        assert result["obp"] == expected_obp
        assert result["pa"] == expected_pa

    def test_missing_rate_in_one_system(self) -> None:
        projections = [
            ({"avg": 0.300, "obp": 0.400, "pa": 600.0}, 0.6),
            ({"avg": 0.250, "pa": 500.0}, 0.4),  # missing obp
        ]
        result = blend_rates(projections, rate_stats=["avg", "obp"], pt_stat="pa")
        expected_avg = (0.300 * 0.6 + 0.250 * 0.4) / (0.6 + 0.4)
        assert result["avg"] == expected_avg
        # obp only from one system
        assert result["obp"] == 0.400

    def test_empty_projections(self) -> None:
        result = blend_rates([], rate_stats=["avg"], pt_stat="pa")
        assert result == {}


class TestWeightedSpread:
    def test_two_systems_equal_weights(self) -> None:
        projections = [
            ({"hr": 30.0}, 1.0),
            ({"hr": 20.0}, 1.0),
        ]
        result = weighted_spread(projections, stats=["hr"])
        assert "hr" in result
        dist = result["hr"]
        assert isinstance(dist, StatDistribution)
        assert dist.stat == "hr"
        # Equal weights: mean = 25, values at 20 and 30
        assert dist.mean == 25.0
        # p50 is the median (midpoint with 2 equal-weight values)
        assert dist.p50 == 25.0
        # With 2 equal-weight values, CDF midpoints are at 0.25 and 0.75.
        # Percentiles outside that range clamp to the boundary values.
        assert dist.p10 == 20.0  # below CDF midpoint 0.25 → clamp to min
        assert dist.p25 == 20.0  # at first CDF midpoint
        assert dist.p75 == 30.0  # at second CDF midpoint
        assert dist.p90 == 30.0  # above CDF midpoint 0.75 → clamp to max
        # std: sqrt(sum(w*(v-mean)^2) / sum(w)) = sqrt((1*(20-25)^2 + 1*(30-25)^2)/2) = 5.0
        assert dist.std is not None
        assert math.isclose(dist.std, 5.0)  # narrowed by assert above

    def test_two_systems_unequal_weights(self) -> None:
        projections = [
            ({"hr": 30.0}, 0.7),
            ({"hr": 20.0}, 0.3),
        ]
        result = weighted_spread(projections, stats=["hr"])
        dist = result["hr"]
        # Weighted mean: (30*0.7 + 20*0.3) / 1.0 = 27.0
        assert dist.mean is not None
        assert math.isclose(dist.mean, 27.0)
        # With unequal weights, percentiles shift toward heavier system (30)
        # The CDF midpoint (p50) should be > 25 (the unweighted midpoint)
        assert dist.p50 > 25.0

    def test_multiple_stats(self) -> None:
        projections = [
            ({"hr": 30.0, "rbi": 100.0}, 1.0),
            ({"hr": 20.0, "rbi": 80.0}, 1.0),
        ]
        result = weighted_spread(projections, stats=["hr", "rbi"])
        assert "hr" in result
        assert "rbi" in result
        assert result["hr"].stat == "hr"
        assert result["rbi"].stat == "rbi"
        assert result["hr"].mean == 25.0
        assert result["rbi"].mean == 90.0

    def test_single_system_returns_empty(self) -> None:
        projections = [
            ({"hr": 30.0}, 1.0),
        ]
        result = weighted_spread(projections, stats=["hr"])
        assert result == {}

    def test_stat_missing_in_some_systems(self) -> None:
        projections = [
            ({"hr": 30.0, "rbi": 100.0}, 1.0),
            ({"hr": 20.0}, 1.0),  # no rbi
        ]
        result = weighted_spread(projections, stats=["hr", "rbi"])
        # hr has 2 systems → distribution computed
        assert "hr" in result
        # rbi has only 1 system → skipped
        assert "rbi" not in result

    def test_uniform_weights_match_simple_quantiles(self) -> None:
        """With many equal-weight systems, percentiles approximate simple quantiles."""
        values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]
        projections = [({"hr": v}, 1.0) for v in values]
        result = weighted_spread(projections, stats=["hr"])
        dist = result["hr"]
        # Mean should be average of all values
        assert dist.mean is not None
        assert math.isclose(dist.mean, 32.5)
        # Median (p50) should be close to 32.5
        assert math.isclose(dist.p50, 32.5, abs_tol=0.1)
