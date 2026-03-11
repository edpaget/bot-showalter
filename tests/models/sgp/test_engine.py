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


class TestRunSgpPipelineOptimal:
    def test_optimal_assignment_default(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 2}
        result = run_sgp_pipeline(stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0)
        assert result.assignments is not None
        assert len(result.assignments) == 2

    def test_greedy_flag_disables_assignments(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        result = run_sgp_pipeline(
            stats,
            categories,
            denominators,
            positions,
            roster_spots,
            num_teams=1,
            budget=100.0,
            use_optimal_assignment=False,
        )
        assert result.assignments is None

    def test_optimal_dollar_sum(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 3}
        result = run_sgp_pipeline(stats, categories, denominators, positions, roster_spots, num_teams=1, budget=90.0)
        assert sum(result.dollar_values) == pytest.approx(90.0)


class TestVolumeWeighted:
    def test_volume_weighted_false_matches_default(self) -> None:
        """volume_weighted=False should produce identical results to default."""
        stats = [
            {"er": 60.0, "ip": 200.0},
            {"er": 30.0, "ip": 100.0},
        ]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}
        result_default = compute_sgp_scores(stats, categories, denominators)
        result_off = compute_sgp_scores(stats, categories, denominators, volume_weighted=False)
        for a, b in zip(result_default, result_off, strict=True):
            assert a.category_sgp == b.category_sgp
            assert a.composite_sgp == pytest.approx(b.composite_sgp)

    def test_rate_stat_scales_with_ip(self) -> None:
        """Same ERA, IP 180 vs 60 → ~3:1 SGP ratio when volume_weighted=True."""
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}
        # Use a 3-player pool so baseline differs from the two equal-rate players.
        stats_3 = [
            {"er": 36.0, "ip": 180.0},  # rate = 0.20 (good)
            {"er": 12.0, "ip": 60.0},  # rate = 0.20 (good, same rate)
            {"er": 60.0, "ip": 200.0},  # rate = 0.30 (bad, sets baseline)
        ]
        result_3 = compute_sgp_scores(stats_3, categories, denominators, volume_weighted=True)
        # Both player 0 and 1 have same rate, but player 0 has 3x the IP
        sgp_0 = result_3[0].category_sgp["era"]
        sgp_1 = result_3[1].category_sgp["era"]
        assert sgp_0 != 0.0
        assert sgp_1 != 0.0
        assert sgp_0 / sgp_1 == pytest.approx(180.0 / 60.0)

    def test_volume_weighted_baseline_is_weighted_mean(self) -> None:
        """Baseline should be volume-weighted mean, not median, when enabled."""
        # Skewed pool: one high-IP pitcher with low rate, two low-IP with high rate
        stats = [
            {"er": 60.0, "ip": 300.0},  # rate = 0.20
            {"er": 30.0, "ip": 100.0},  # rate = 0.30
            {"er": 30.0, "ip": 100.0},  # rate = 0.30
        ]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}

        result_default = compute_sgp_scores(stats, categories, denominators, volume_weighted=False)
        result_weighted = compute_sgp_scores(stats, categories, denominators, volume_weighted=True)

        # Median rate = 0.30 (middle of [0.20, 0.30, 0.30])
        # Weighted mean = (0.20*300 + 0.30*100 + 0.30*100) / (300+100+100) = 120/500 = 0.24
        # Player 0 gets different baseline → different raw SGP
        assert result_default[0].category_sgp["era"] != pytest.approx(result_weighted[0].category_sgp["era"])

    def test_volume_weighted_counting_stats_unaffected(self) -> None:
        """Counting stats should be identical regardless of volume_weighted flag."""
        stats = [{"hr": 30.0, "ip": 200.0}, {"hr": 20.0, "ip": 60.0}]
        categories = [_counting("hr")]
        denominators = {"hr": 10.0}
        result_off = compute_sgp_scores(stats, categories, denominators, volume_weighted=False)
        result_on = compute_sgp_scores(stats, categories, denominators, volume_weighted=True)
        for a, b in zip(result_off, result_on, strict=True):
            assert a.category_sgp == b.category_sgp

    def test_volume_weighted_zero_ip_gives_zero(self) -> None:
        """Player with zero IP should get zero SGP for rate stat."""
        stats = [
            {"er": 0.0, "ip": 0.0},
            {"er": 30.0, "ip": 100.0},
        ]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}
        result = compute_sgp_scores(stats, categories, denominators, volume_weighted=True)
        assert result[0].category_sgp["era"] == 0.0

    def test_volume_weighted_single_player(self) -> None:
        """Single player: weight = 1.0, raw SGP = 0 (baseline = own rate)."""
        stats = [{"er": 30.0, "ip": 100.0}]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        denominators = {"era": 0.1}
        result = compute_sgp_scores(stats, categories, denominators, volume_weighted=True)
        assert result[0].category_sgp["era"] == pytest.approx(0.0)

    def test_volume_weighted_higher_is_better(self) -> None:
        """OBP with PA weighting — higher is better direction."""
        stats = [
            {"obp_num": 210.0, "pa": 600.0},  # OBP = 0.350
            {"obp_num": 60.0, "pa": 200.0},  # OBP = 0.300
        ]
        categories = [_rate("obp", "obp_num", "pa", Direction.HIGHER)]
        denominators = {"obp": 0.005}
        result = compute_sgp_scores(stats, categories, denominators, volume_weighted=True)
        # weighted baseline = (0.350*600 + 0.300*200) / 800 = 270/800 = 0.3375
        # avg_vol = mean(600, 200) = 400
        # Player 0: raw = (0.350 - 0.3375) / 0.005 = 2.5, weight = 600/400 = 1.5 → 3.75
        # Player 1: raw = (0.300 - 0.3375) / 0.005 = -7.5, weight = 200/400 = 0.5 → -3.75
        assert result[0].category_sgp["obp"] > 0.0
        assert result[1].category_sgp["obp"] < 0.0
        assert result[0].category_sgp["obp"] == pytest.approx(3.75)
        assert result[1].category_sgp["obp"] == pytest.approx(-3.75)


class TestPipelineVolumeWeighted:
    def test_pipeline_with_volume_weighted(self) -> None:
        """run_sgp_pipeline with volume_weighted=True produces different composite scores."""
        stats = [
            {"er": 36.0, "ip": 180.0, "w": 15.0},
            {"er": 12.0, "ip": 60.0, "w": 5.0},
            {"er": 60.0, "ip": 200.0, "w": 12.0},
        ]
        categories = [
            _rate("era", "er", "ip", Direction.LOWER),
            _counting("w"),
        ]
        denominators = {"era": 0.1, "w": 3.0}
        positions = [["P"], ["P"], ["P"]]
        roster_spots = {"P": 3}
        result_off = run_sgp_pipeline(
            stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0, volume_weighted=False
        )
        result_on = run_sgp_pipeline(
            stats, categories, denominators, positions, roster_spots, num_teams=1, budget=100.0, volume_weighted=True
        )
        sgp_off = [s.composite_sgp for s in result_off.sgp_scores]
        sgp_on = [s.composite_sgp for s in result_on.sgp_scores]
        assert sgp_off != sgp_on
