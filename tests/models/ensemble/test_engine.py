from fantasy_baseball_manager.models.ensemble.engine import blend_rates, weighted_average


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
