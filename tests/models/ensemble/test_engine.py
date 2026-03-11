import math

import pytest

from fantasy_baseball_manager.domain.projection import StatDistribution
from fantasy_baseball_manager.models.ensemble.engine import (
    blend_rates,
    normalize_routed_counting_stats,
    per_stat_weighted,
    routed,
    weighted_average,
    weighted_spread,
)


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


class TestBlendRatesWithConsensusPT:
    def test_consensus_pt_overrides_averaged_pt(self) -> None:
        projections = [
            ({"avg": 0.300, "obp": 0.400, "pa": 600.0}, 0.6),
            ({"avg": 0.250, "obp": 0.350, "pa": 500.0}, 0.4),
        ]
        result = blend_rates(projections, rate_stats=["avg", "obp"], pt_stat="pa", consensus_pt=550.0)
        assert result["pa"] == 550.0
        # Rates still weight-averaged
        expected_avg = (0.300 * 0.6 + 0.250 * 0.4) / (0.6 + 0.4)
        expected_obp = (0.400 * 0.6 + 0.350 * 0.4) / (0.6 + 0.4)
        assert result["avg"] == expected_avg
        assert result["obp"] == expected_obp

    def test_consensus_pt_none_preserves_current_behavior(self) -> None:
        projections = [
            ({"avg": 0.300, "obp": 0.400, "pa": 600.0}, 0.6),
            ({"avg": 0.250, "obp": 0.350, "pa": 500.0}, 0.4),
        ]
        result = blend_rates(projections, rate_stats=["avg", "obp"], pt_stat="pa", consensus_pt=None)
        expected_pa = (600.0 * 0.6 + 500.0 * 0.4) / (0.6 + 0.4)
        assert result["pa"] == expected_pa

    def test_consensus_pt_with_pitcher_ip(self) -> None:
        projections = [
            ({"era": 3.50, "whip": 1.20, "ip": 180.0}, 0.5),
            ({"era": 3.80, "whip": 1.25, "ip": 160.0}, 0.5),
        ]
        result = blend_rates(projections, rate_stats=["era", "whip"], pt_stat="ip", consensus_pt=170.0)
        assert result["ip"] == 170.0
        # Rates still averaged
        assert result["era"] == (3.50 * 0.5 + 3.80 * 0.5) / 1.0
        assert result["whip"] == (1.20 * 0.5 + 1.25 * 0.5) / 1.0


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


class TestRouted:
    def test_basic_routing(self) -> None:
        system_stats = {
            "statcast-gbm": {"obp": 0.350, "avg": 0.280},
            "steamer": {"hr": 30.0, "rbi": 100.0, "obp": 0.340},
        }
        routes = {"obp": "statcast-gbm", "hr": "steamer"}
        result = routed(system_stats, routes)
        assert result == {"obp": 0.350, "hr": 30.0}

    def test_fallback_when_primary_missing(self) -> None:
        system_stats = {
            "statcast-gbm": {"obp": 0.350},
            "steamer": {"hr": 30.0, "obp": 0.340},
        }
        routes = {"hr": "statcast-gbm", "obp": "statcast-gbm"}
        result = routed(system_stats, routes, fallback="steamer")
        assert result["obp"] == 0.350  # from primary
        assert result["hr"] == 30.0  # from fallback

    def test_missing_stat_no_fallback(self) -> None:
        system_stats = {
            "statcast-gbm": {"obp": 0.350},
            "steamer": {"hr": 30.0},
        }
        routes = {"hr": "statcast-gbm"}
        result = routed(system_stats, routes)
        # statcast-gbm lacks hr, no fallback → omitted
        assert result == {}

    def test_missing_stat_in_all_systems(self) -> None:
        system_stats = {
            "statcast-gbm": {"obp": 0.350},
            "steamer": {"hr": 30.0},
        }
        routes = {"sb": "statcast-gbm"}
        result = routed(system_stats, routes, fallback="steamer")
        # Neither system has sb → omitted
        assert result == {}

    def test_system_not_in_system_stats(self) -> None:
        system_stats = {
            "steamer": {"hr": 30.0, "obp": 0.340},
        }
        routes = {"hr": "statcast-gbm", "obp": "steamer"}
        result = routed(system_stats, routes, fallback="steamer")
        # statcast-gbm not present → fallback to steamer for hr
        assert result == {"hr": 30.0, "obp": 0.340}


class TestPerStatWeighted:
    def test_different_weights_per_stat(self) -> None:
        system_stats = {
            "statcast-gbm": {"obp": 0.350, "hr": 25.0},
            "steamer": {"obp": 0.330, "hr": 30.0},
        }
        stat_weights = {
            "obp": {"statcast-gbm": 0.7, "steamer": 0.3},
            "hr": {"statcast-gbm": 0.0, "steamer": 1.0},
        }
        result = per_stat_weighted(system_stats, stat_weights)
        expected_obp = (0.350 * 0.7 + 0.330 * 0.3) / (0.7 + 0.3)
        assert result["obp"] == expected_obp
        assert result["hr"] == 30.0  # 100% steamer

    def test_missing_system_for_stat(self) -> None:
        system_stats = {
            "steamer": {"obp": 0.330},
        }
        stat_weights = {
            "obp": {"statcast-gbm": 0.7, "steamer": 0.3},
        }
        result = per_stat_weighted(system_stats, stat_weights)
        # statcast-gbm not in system_stats → excluded, only steamer
        assert result["obp"] == 0.330

    def test_stat_in_single_system(self) -> None:
        system_stats = {
            "statcast-gbm": {"obp": 0.350},
            "steamer": {"hr": 30.0},
        }
        stat_weights = {
            "obp": {"statcast-gbm": 1.0},
        }
        result = per_stat_weighted(system_stats, stat_weights)
        assert result == {"obp": 0.350}

    def test_empty_stat_weights(self) -> None:
        system_stats = {
            "statcast-gbm": {"obp": 0.350},
        }
        result = per_stat_weighted(system_stats, stat_weights={})
        assert result == {}


class TestNormalizeRoutedCountingStats:
    """Tests for post-routing normalization of counting stats."""

    _PITCHER_COUNTING = frozenset({"w", "l", "g", "gs", "sv", "hld", "ip", "h", "er", "hr", "bb", "so"})

    def test_ip_routed_differently_scales_counting_stats(self) -> None:
        """IP from playing_time (160), counting stats from steamer (200 IP) → scale by 0.8."""
        result_stats = {"ip": 160.0, "er": 80.0, "so": 200.0, "era": 3.60}
        routes = {"ip": "playing_time", "er": "steamer", "so": "steamer", "era": "steamer"}
        system_stats = {
            "playing_time": {"ip": 160.0},
            "steamer": {"ip": 200.0, "er": 80.0, "so": 200.0, "era": 3.60},
        }
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result["ip"] == 160.0
        assert result["er"] == pytest.approx(80.0 * 160.0 / 200.0)
        assert result["so"] == pytest.approx(200.0 * 160.0 / 200.0)
        # Rate stat unchanged
        assert result["era"] == 3.60

    def test_all_stats_same_system_no_scaling(self) -> None:
        """When all stats come from the same system, no normalization needed."""
        result_stats = {"ip": 200.0, "er": 80.0, "so": 200.0}
        routes = {"ip": "steamer", "er": "steamer", "so": "steamer"}
        system_stats = {"steamer": {"ip": 200.0, "er": 80.0, "so": 200.0}}
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result == result_stats

    def test_ip_not_in_routes_no_scaling(self) -> None:
        """When IP is not explicitly routed, no normalization."""
        result_stats = {"ip": 200.0, "er": 80.0}
        routes = {"er": "steamer"}
        system_stats = {"steamer": {"ip": 200.0, "er": 80.0}}
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result == result_stats

    def test_source_system_zero_ip_no_scaling(self) -> None:
        """When source system has 0 IP, skip scaling to avoid div/0."""
        result_stats = {"ip": 160.0, "er": 80.0}
        routes = {"ip": "playing_time", "er": "steamer"}
        system_stats = {
            "playing_time": {"ip": 160.0},
            "steamer": {"ip": 0.0, "er": 80.0},
        }
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result["ip"] == 160.0
        assert result["er"] == 80.0  # unchanged

    def test_rate_stats_left_unchanged(self) -> None:
        """Stats not in counting_stats are never scaled."""
        result_stats = {"ip": 160.0, "era": 3.60, "whip": 1.20, "er": 80.0}
        routes = {"ip": "playing_time", "era": "steamer", "whip": "steamer", "er": "steamer"}
        system_stats = {
            "playing_time": {"ip": 160.0},
            "steamer": {"ip": 200.0, "era": 3.60, "whip": 1.20, "er": 80.0},
        }
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result["era"] == 3.60
        assert result["whip"] == 1.20
        assert result["er"] == pytest.approx(80.0 * 160.0 / 200.0)

    def test_batter_pa_routed_differently(self) -> None:
        """Batter variant: PA from one system, counting stats from another."""
        batter_counting = frozenset({"pa", "ab", "h", "hr", "rbi", "r", "sb", "bb", "so"})
        result_stats = {"pa": 500.0, "hr": 30.0, "rbi": 100.0, "avg": 0.280}
        routes = {"pa": "playing_time", "hr": "steamer", "rbi": "steamer", "avg": "steamer"}
        system_stats = {
            "playing_time": {"pa": 500.0},
            "steamer": {"pa": 600.0, "hr": 30.0, "rbi": 100.0, "avg": 0.280},
        }
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="pa",
            counting_stats=batter_counting,
        )
        assert result["pa"] == 500.0
        assert result["hr"] == pytest.approx(30.0 * 500.0 / 600.0)
        assert result["rbi"] == pytest.approx(100.0 * 500.0 / 600.0)
        assert result["avg"] == 0.280  # rate stat, not in counting_stats

    def test_does_not_mutate_input(self) -> None:
        """The function returns a new dict, not mutating the input."""
        result_stats = {"ip": 160.0, "er": 80.0}
        routes = {"ip": "playing_time", "er": "steamer"}
        system_stats = {
            "playing_time": {"ip": 160.0},
            "steamer": {"ip": 200.0, "er": 80.0},
        }
        original = dict(result_stats)
        normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result_stats == original

    def test_pt_stat_missing_from_result_no_scaling(self) -> None:
        """When pt_stat is not in result_stats, return unchanged."""
        result_stats = {"er": 80.0, "so": 200.0}
        routes = {"er": "steamer", "so": "steamer"}
        system_stats = {"steamer": {"ip": 200.0, "er": 80.0, "so": 200.0}}
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result == result_stats

    def test_pt_stat_zero_no_scaling(self) -> None:
        """When routed PT is 0, return unchanged."""
        result_stats = {"ip": 0.0, "er": 80.0}
        routes = {"ip": "playing_time", "er": "steamer"}
        system_stats = {
            "playing_time": {"ip": 0.0},
            "steamer": {"ip": 200.0, "er": 80.0},
        }
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result == result_stats

    def test_counting_stat_not_routed_skipped(self) -> None:
        """Counting stat without an explicit route is not scaled."""
        result_stats = {"ip": 160.0, "er": 80.0, "so": 200.0}
        routes = {"ip": "playing_time", "er": "steamer"}  # so not in routes
        system_stats = {
            "playing_time": {"ip": 160.0},
            "steamer": {"ip": 200.0, "er": 80.0, "so": 200.0},
        }
        result = normalize_routed_counting_stats(
            result_stats,
            routes,
            system_stats,
            pt_stat="ip",
            counting_stats=self._PITCHER_COUNTING,
        )
        assert result["er"] == pytest.approx(80.0 * 0.8)
        assert result["so"] == 200.0  # not in routes → not scaled
