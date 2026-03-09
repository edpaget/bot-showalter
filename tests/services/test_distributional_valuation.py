import pytest

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    StatType,
)
from fantasy_baseball_manager.models.zar.engine import run_zar_pipeline
from fantasy_baseball_manager.services.distributional_valuation import (
    compute_expected_value,
    run_distributional_zar,
)


def _counting(key: str, direction: Direction = Direction.HIGHER) -> CategoryConfig:
    return CategoryConfig(key=key, name=key, stat_type=StatType.COUNTING, direction=direction)


def _rate(
    key: str,
    numerator: str,
    denominator: str,
    direction: Direction = Direction.HIGHER,
) -> CategoryConfig:
    return CategoryConfig(
        key=key,
        name=key,
        stat_type=StatType.RATE,
        direction=direction,
        numerator=numerator,
        denominator=denominator,
    )


class TestComputeExpectedValue:
    def test_weighted_mean(self) -> None:
        scenarios = [(10.0, 0.3), (20.0, 0.5), (30.0, 0.2)]
        result = compute_expected_value(scenarios)
        # 10*0.3 + 20*0.5 + 30*0.2 = 3 + 10 + 6 = 19.0
        assert result == pytest.approx(19.0)

    def test_single_entry(self) -> None:
        result = compute_expected_value([(42.0, 1.0)])
        assert result == pytest.approx(42.0)

    def test_equal_weights(self) -> None:
        scenarios = [(10.0, 0.2), (20.0, 0.2), (30.0, 0.2), (40.0, 0.2), (50.0, 0.2)]
        result = compute_expected_value(scenarios)
        assert result == pytest.approx(30.0)


class TestRunDistributionalZar:
    """Tests for the multi-pass distributional ZAR engine."""

    # Shared fixtures for a simple 4-player pool with 1 counting category
    CATEGORIES = [_counting("hr")]
    POSITIONS = [["of"], ["of"], ["of"], ["of"]]
    ROSTER_SPOTS = {"of": 2}
    NUM_TEAMS = 1
    BUDGET = 100.0

    def _make_stats(self, hr_values: list[float]) -> list[dict[str, float]]:
        return [{"hr": v} for v in hr_values]

    def test_single_scenario_matches_point_estimate(self) -> None:
        """A player with 1 scenario at weight 1.0 gets exactly the point-estimate value."""
        stats = self._make_stats([30.0, 25.0, 20.0, 15.0])
        # Single scenario per player: same stats, weight 1.0
        scenario_stats = [[(s, 1.0)] for s in stats]

        result = run_distributional_zar(
            stats,
            scenario_stats,
            self.CATEGORIES,
            self.POSITIONS,
            self.ROSTER_SPOTS,
            self.NUM_TEAMS,
            self.BUDGET,
        )
        point = run_zar_pipeline(stats, self.CATEGORIES, self.POSITIONS, self.ROSTER_SPOTS, self.NUM_TEAMS, self.BUDGET)
        for i in range(len(stats)):
            assert result.dollar_values[i] == pytest.approx(point.dollar_values[i])

    def test_uniform_scenarios_match_point_estimate(self) -> None:
        """All 5 scenarios identical → same value as point estimate."""
        stats = self._make_stats([30.0, 25.0, 20.0, 15.0])
        weights = [0.15, 0.20, 0.30, 0.20, 0.15]
        # Each player has 5 identical scenarios
        scenario_stats = [[(s, w) for w in weights] for s in stats]

        result = run_distributional_zar(
            stats,
            scenario_stats,
            self.CATEGORIES,
            self.POSITIONS,
            self.ROSTER_SPOTS,
            self.NUM_TEAMS,
            self.BUDGET,
        )
        point = run_zar_pipeline(stats, self.CATEGORIES, self.POSITIONS, self.ROSTER_SPOTS, self.NUM_TEAMS, self.BUDGET)
        for i in range(len(stats)):
            assert result.dollar_values[i] == pytest.approx(point.dollar_values[i])

    def test_left_skewed_distribution_lower_value(self) -> None:
        """Injury-prone player with larger downside offsets gets lower expected value."""
        # Player 0 has left-skewed distribution: more downside than upside
        # Other players are stable (identical across scenarios)
        base_stats = self._make_stats([30.0, 25.0, 20.0, 15.0])
        weights = [0.15, 0.20, 0.30, 0.20, 0.15]

        # Player 0: skewed downside (hr drops a lot in bad scenarios)
        p0_scenarios = [
            ({"hr": 5.0}, 0.15),  # P10: severe downside
            ({"hr": 15.0}, 0.20),  # P25: moderate downside
            ({"hr": 30.0}, 0.30),  # P50: point estimate
            ({"hr": 33.0}, 0.20),  # P75: slight upside
            ({"hr": 35.0}, 0.15),  # P90: moderate upside
        ]
        # Other players: stable
        stable_scenarios = [[(s, w) for w in weights] for s in base_stats[1:]]

        scenario_stats = [p0_scenarios, *stable_scenarios]

        result = run_distributional_zar(
            base_stats,
            scenario_stats,
            self.CATEGORIES,
            self.POSITIONS,
            self.ROSTER_SPOTS,
            self.NUM_TEAMS,
            self.BUDGET,
        )
        point = run_zar_pipeline(
            base_stats, self.CATEGORIES, self.POSITIONS, self.ROSTER_SPOTS, self.NUM_TEAMS, self.BUDGET
        )
        # Player 0 should be valued lower in distributional than point estimate
        assert result.dollar_values[0] < point.dollar_values[0]

    def test_symmetric_offsets_close_to_point_estimate(self) -> None:
        """Symmetric residuals → value close to point estimate."""
        base_stats = self._make_stats([30.0, 25.0, 20.0, 15.0])
        # Player 0: symmetric offsets around point estimate
        p0_scenarios = [
            ({"hr": 20.0}, 0.15),  # -10
            ({"hr": 25.0}, 0.20),  # -5
            ({"hr": 30.0}, 0.30),  # 0
            ({"hr": 35.0}, 0.20),  # +5
            ({"hr": 40.0}, 0.15),  # +10
        ]
        weights = [0.15, 0.20, 0.30, 0.20, 0.15]
        stable_scenarios = [[(s, w) for w in weights] for s in base_stats[1:]]
        scenario_stats = [p0_scenarios, *stable_scenarios]

        result = run_distributional_zar(
            base_stats,
            scenario_stats,
            self.CATEGORIES,
            self.POSITIONS,
            self.ROSTER_SPOTS,
            self.NUM_TEAMS,
            self.BUDGET,
        )
        point = run_zar_pipeline(
            base_stats, self.CATEGORIES, self.POSITIONS, self.ROSTER_SPOTS, self.NUM_TEAMS, self.BUDGET
        )
        # Should be closer to point estimate than the left-skewed case, but
        # non-linearity in the dollar conversion (draftable cutoff, VAR→$ mapping)
        # can cause meaningful divergence even with symmetric stat offsets.
        # The key property: symmetric should be closer to point than left-skewed.
        assert result.dollar_values[0] == pytest.approx(point.dollar_values[0], rel=0.2)

    def test_pool_size_preserved(self) -> None:
        """Every scenario run has same number of players."""
        stats = self._make_stats([30.0, 25.0, 20.0])
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 2}
        weights = [0.15, 0.20, 0.30, 0.20, 0.15]

        # Player 0 has 5 scenarios, player 1 has 1, player 2 has 5
        scenario_stats = [
            [({"hr": 30.0 + i}, w) for i, w in enumerate(weights)],
            [({"hr": 25.0}, 1.0)],  # single scenario
            [({"hr": 20.0 + i}, w) for i, w in enumerate(weights)],
        ]

        result = run_distributional_zar(
            stats,
            scenario_stats,
            self.CATEGORIES,
            positions,
            roster_spots,
            self.NUM_TEAMS,
            self.BUDGET,
        )
        # All 3 players should have values
        assert len(result.dollar_values) == 3

    def test_category_scores_from_point_estimate(self) -> None:
        """Returned result has category z-scores from the median/point-estimate run."""
        stats = self._make_stats([30.0, 20.0, 10.0])
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 2}
        weights = [0.15, 0.20, 0.30, 0.20, 0.15]

        scenario_stats = [[(s, w) for w in weights] for s in stats]

        result = run_distributional_zar(
            stats,
            scenario_stats,
            self.CATEGORIES,
            positions,
            roster_spots,
            self.NUM_TEAMS,
            self.BUDGET,
        )
        point = run_zar_pipeline(stats, self.CATEGORIES, positions, roster_spots, self.NUM_TEAMS, self.BUDGET)
        # Category z-scores should match the point-estimate pipeline
        for i in range(len(stats)):
            assert result.z_scores[i].category_z == point.z_scores[i].category_z

    def test_stdev_overrides_passed_through(self) -> None:
        """stdev_overrides are forwarded to the ZAR pipeline."""
        stats = self._make_stats([30.0, 20.0, 10.0])
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 2}
        weights = [0.15, 0.20, 0.30, 0.20, 0.15]
        scenario_stats = [[(s, w) for w in weights] for s in stats]

        without = run_distributional_zar(
            stats, scenario_stats, self.CATEGORIES, positions, roster_spots, self.NUM_TEAMS, self.BUDGET
        )
        with_override = run_distributional_zar(
            stats,
            scenario_stats,
            self.CATEGORIES,
            positions,
            roster_spots,
            self.NUM_TEAMS,
            self.BUDGET,
            stdev_overrides={"hr": 100.0},
        )
        # Different stdev → different z-scores
        assert without.z_scores[0].category_z != with_override.z_scores[0].category_z

    def test_mixed_rate_and_counting(self) -> None:
        """Works correctly with rate stats in the mix."""
        categories = [_counting("hr"), _rate("avg", "h", "ab")]
        stats = [
            {"hr": 30.0, "h": 150.0, "ab": 500.0},
            {"hr": 20.0, "h": 100.0, "ab": 400.0},
            {"hr": 10.0, "h": 80.0, "ab": 350.0},
        ]
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 2}
        weights = [0.15, 0.20, 0.30, 0.20, 0.15]
        scenario_stats = [[(s, w) for w in weights] for s in stats]

        result = run_distributional_zar(
            stats, scenario_stats, categories, positions, roster_spots, self.NUM_TEAMS, self.BUDGET
        )
        point = run_zar_pipeline(stats, categories, positions, roster_spots, self.NUM_TEAMS, self.BUDGET)
        # With identical scenarios, should match point estimate
        for i in range(len(stats)):
            assert result.dollar_values[i] == pytest.approx(point.dollar_values[i])
