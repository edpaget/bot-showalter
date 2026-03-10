import pytest

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    StatType,
)
from fantasy_baseball_manager.models.sgp.engine import (
    SgpPipelineResult,
    compute_sgp_scores,
    run_sgp_pipeline,
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


class TestComputeSgpScores:
    def test_counting_stat_with_known_denominator(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        result = compute_sgp_scores(stats, categories, denominators)
        assert result[0].category_sgp["hr"] == pytest.approx(3.0)  # 30/10
        assert result[1].category_sgp["hr"] == pytest.approx(2.0)  # 20/10

    def test_counting_stat_composite_key(self) -> None:
        stats = [{"sv": 30.0, "hld": 20.0}]
        categories = [_counting("sv+hld")]
        denominators = {"sv+hld": 10.0}
        result = compute_sgp_scores(stats, categories, denominators)
        assert result[0].category_sgp["sv+hld"] == pytest.approx(5.0)  # 50/10

    def test_rate_stat_independent_of_ip(self) -> None:
        """Rate stat SGP depends only on ERA, not on IP — the key difference from ZAR."""
        stats = [
            {"er": 60.0, "ip": 200.0},  # ERA = 3.0
            {"er": 30.0, "ip": 100.0},  # ERA = 3.0
        ]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}
        result = compute_sgp_scores(stats, categories, denominators)
        # Both have same ERA → same SGP (median baseline = 3.0, diff = 0)
        assert result[0].category_sgp["era"] == pytest.approx(0.0)
        assert result[1].category_sgp["era"] == pytest.approx(0.0)

    def test_rate_stat_different_era_same_ip(self) -> None:
        """Different ERA should produce different SGP scores."""
        stats = [
            {"er": 60.0, "ip": 200.0},  # rate = 0.3
            {"er": 80.0, "ip": 200.0},  # rate = 0.4
        ]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}
        result = compute_sgp_scores(stats, categories, denominators)
        # Median rate = 0.35, denom = 0.1
        # Player 0: (0.35 - 0.3) / 0.1 = 0.5
        # Player 1: (0.35 - 0.4) / 0.1 = -0.5
        assert result[0].category_sgp["era"] == pytest.approx(0.5)
        assert result[1].category_sgp["era"] == pytest.approx(-0.5)

    def test_rate_stat_higher_is_better(self) -> None:
        stats = [
            {"h": 150.0, "ab": 500.0},  # AVG = 0.300
            {"h": 125.0, "ab": 500.0},  # AVG = 0.250
        ]
        categories = [_rate("avg", "h", "ab")]
        denominators = {"avg": 0.005}
        result = compute_sgp_scores(stats, categories, denominators)
        # Median AVG = 0.275, denom = 0.005
        # Player 0: (0.300 - 0.275) / 0.005 = 5.0
        # Player 1: (0.250 - 0.275) / 0.005 = -5.0
        assert result[0].category_sgp["avg"] == pytest.approx(5.0)
        assert result[1].category_sgp["avg"] == pytest.approx(-5.0)

    def test_composite_is_sum_of_category_sgp(self) -> None:
        stats = [{"hr": 30.0, "r": 100.0}, {"hr": 20.0, "r": 80.0}]
        categories = [_counting("hr"), _counting("r")]
        denominators = {"hr": 10.0, "r": 20.0}
        result = compute_sgp_scores(stats, categories, denominators)
        for s in result:
            expected = sum(s.category_sgp.values())
            assert s.composite_sgp == pytest.approx(expected)

    def test_zero_denominator_gives_zero_sgp(self) -> None:
        stats = [{"hr": 30.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 0.0}
        result = compute_sgp_scores(stats, categories, denominators)
        assert result[0].category_sgp["hr"] == 0.0

    def test_missing_denominator_gives_zero_sgp(self) -> None:
        stats = [{"hr": 30.0}]
        categories = [_counting("hr")]
        denominators = {}
        result = compute_sgp_scores(stats, categories, denominators)
        assert result[0].category_sgp["hr"] == 0.0

    def test_empty_pool(self) -> None:
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        result = compute_sgp_scores([], categories, denominators)
        assert result == []

    def test_rate_stat_zero_denom_value_gives_zero(self) -> None:
        stats = [{"er": 0.0, "ip": 0.0}]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}
        result = compute_sgp_scores(stats, categories, denominators)
        assert result[0].category_sgp["era"] == 0.0

    def test_player_index_matches_position(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        result = compute_sgp_scores(stats, categories, denominators)
        assert result[0].player_index == 0
        assert result[1].player_index == 1


class TestRunSgpPipeline:
    def test_returns_pipeline_result(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        result = run_sgp_pipeline(stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0)
        assert isinstance(result, SgpPipelineResult)

    def test_empty_input(self) -> None:
        result = run_sgp_pipeline([], [_counting("hr")], {"hr": 10.0}, [], {}, num_teams=1, budget=100.0)
        assert result.sgp_scores == []
        assert result.replacement == {}
        assert result.dollar_values == []

    def test_dollar_values_sum_to_budget(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 3}
        result = run_sgp_pipeline(stats, categories, denominators, positions, roster_spots, num_teams=1, budget=90.0)
        draftable_sum = sum(d for d in result.dollar_values if d > 0.0)
        assert draftable_sum == pytest.approx(90.0)

    def test_sgp_scores_length_matches_input(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        result = run_sgp_pipeline(stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0)
        assert len(result.sgp_scores) == 2
        assert len(result.dollar_values) == 2

    def test_replacement_keys_match_roster_spots(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        positions = [["c", "of"], ["of"]]
        roster_spots = {"c": 1, "of": 1}
        result = run_sgp_pipeline(stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0)
        assert set(result.replacement.keys()) == {"c", "of"}

    def test_full_pipeline_rate_stat_ip_independent(self) -> None:
        """Two pitchers with same ERA but different IP should get same ERA SGP."""
        stats = [
            {"er": 60.0, "ip": 200.0, "w": 15.0},  # ERA=3.0
            {"er": 30.0, "ip": 100.0, "w": 8.0},  # ERA=3.0
        ]
        categories = [
            _rate("era", "er", "ip", Direction.LOWER),
            _counting("w"),
        ]
        denominators = {"era": 0.1, "w": 3.0}
        positions = [["P"], ["P"]]
        roster_spots = {"P": 2}
        result = run_sgp_pipeline(stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0)
        # ERA SGP should be identical
        assert result.sgp_scores[0].category_sgp["era"] == pytest.approx(result.sgp_scores[1].category_sgp["era"])


class TestComputeSgpScoresDirectRates:
    def test_direct_rate_used_when_present(self) -> None:
        """When stats dict has 'avg' field that disagrees with h/ab, direct rate wins."""
        stats = [
            {"h": 150.0, "ab": 500.0, "avg": 0.350},  # h/ab=0.300, but avg=0.350
            {"h": 125.0, "ab": 500.0, "avg": 0.200},  # h/ab=0.250, but avg=0.200
        ]
        categories = [_rate("avg", "h", "ab")]
        denominators = {"avg": 0.005}
        result_direct = compute_sgp_scores(stats, categories, denominators, use_direct_rates=True)
        result_derived = compute_sgp_scores(stats, categories, denominators, use_direct_rates=False)
        # Results should differ
        assert result_direct[0].category_sgp["avg"] != pytest.approx(result_derived[0].category_sgp["avg"])

    def test_median_baseline_uses_direct_rates(self) -> None:
        """Baseline should be median of direct rates, not derived rates."""
        # Three players with direct avg: 0.300, 0.280, 0.260 → median = 0.280
        # Derived avg would be: 0.250, 0.250, 0.250 → median = 0.250
        stats = [
            {"h": 125.0, "ab": 500.0, "avg": 0.300},
            {"h": 125.0, "ab": 500.0, "avg": 0.280},
            {"h": 125.0, "ab": 500.0, "avg": 0.260},
        ]
        categories = [_rate("avg", "h", "ab")]
        denominators = {"avg": 0.005}
        result = compute_sgp_scores(stats, categories, denominators, use_direct_rates=True)
        # Baseline = median(0.300, 0.280, 0.260) = 0.280
        # Player 0: (0.300 - 0.280) / 0.005 = 4.0
        # Player 2: (0.260 - 0.280) / 0.005 = -4.0
        assert result[0].category_sgp["avg"] == pytest.approx(4.0)
        assert result[2].category_sgp["avg"] == pytest.approx(-4.0)

    def test_fallback_when_key_missing(self) -> None:
        """Player without direct rate key falls back to derived calculation."""
        stats = [
            {"h": 150.0, "ab": 500.0, "avg": 0.350},  # has direct rate
            {"h": 125.0, "ab": 500.0},  # no avg key → derived = 0.250
        ]
        categories = [_rate("avg", "h", "ab")]
        denominators = {"avg": 0.005}
        result = compute_sgp_scores(stats, categories, denominators, use_direct_rates=True)
        # Baseline = median(0.350, 0.250) = 0.300
        # Player 0: (0.350 - 0.300) / 0.005 = 10.0
        # Player 1: (0.250 - 0.300) / 0.005 = -10.0  (derived rate used)
        assert result[0].category_sgp["avg"] == pytest.approx(10.0)
        assert result[1].category_sgp["avg"] == pytest.approx(-10.0)

    def test_false_preserves_behavior(self) -> None:
        """Explicit use_direct_rates=False matches default behavior."""
        stats = [
            {"h": 150.0, "ab": 500.0, "avg": 0.350},
            {"h": 125.0, "ab": 500.0, "avg": 0.200},
        ]
        categories = [_rate("avg", "h", "ab")]
        denominators = {"avg": 0.005}
        result_default = compute_sgp_scores(stats, categories, denominators)
        result_false = compute_sgp_scores(stats, categories, denominators, use_direct_rates=False)
        assert result_default[0].category_sgp == result_false[0].category_sgp
        assert result_default[1].category_sgp == result_false[1].category_sgp

    def test_pipeline_with_direct_rates(self) -> None:
        """run_sgp_pipeline with use_direct_rates=True produces different scores."""
        stats = [
            {"hr": 30.0, "h": 150.0, "ab": 500.0, "avg": 0.350},
            {"hr": 20.0, "h": 100.0, "ab": 400.0, "avg": 0.200},
        ]
        categories = [_counting("hr"), _rate("avg", "h", "ab")]
        denominators = {"hr": 5.0, "avg": 0.005}
        positions = [["of"], ["of"]]
        roster_spots = {"of": 2}
        result_off = run_sgp_pipeline(
            stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0, use_direct_rates=False
        )
        result_on = run_sgp_pipeline(
            stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0, use_direct_rates=True
        )
        sgp_off = [s.composite_sgp for s in result_off.sgp_scores]
        sgp_on = [s.composite_sgp for s in result_on.sgp_scores]
        assert sgp_off != sgp_on
