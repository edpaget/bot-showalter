import statistics

import pytest

from fantasy_baseball_manager.domain.league_settings import (
    BudgetSplitMode,
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.models.zar.engine import (
    PlayerZScores,
    ZarPipelineResult,
    assignment_to_dollars,
    compute_budget_split,
    compute_replacement_level,
    compute_var,
    compute_z_scores,
    convert_rate_stats,
    normalize_composite_z,
    resolve_numerator,
    run_optimal_pipeline,
    run_zar_pipeline,
    var_to_dollars,
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


class TestResolveNumerator:
    def test_single_stat(self) -> None:
        assert resolve_numerator("hr", {"hr": 30.0, "ab": 500.0}) == 30.0

    def test_compound_expression(self) -> None:
        stats = {"bb": 50.0, "h": 150.0}
        assert resolve_numerator("bb+h", stats) == 200.0

    def test_whitespace_in_expression(self) -> None:
        stats = {"bb": 50.0, "h": 150.0}
        assert resolve_numerator("bb + h", stats) == 200.0

    def test_missing_stat_defaults_to_zero(self) -> None:
        stats = {"bb": 50.0}
        assert resolve_numerator("bb+h", stats) == 50.0

    def test_all_missing_returns_zero(self) -> None:
        assert resolve_numerator("bb+h", {}) == 0.0


class TestConvertRateStats:
    def test_counting_stat_unchanged(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        result = convert_rate_stats(stats, categories)
        assert result[0]["hr"] == 30.0
        assert result[1]["hr"] == 20.0

    def test_rate_stat_higher_avg(self) -> None:
        # Player A: .300 in 500 AB, Player B: .250 in 400 AB
        # Baseline = (150+100)/(500+400) = 250/900 ≈ 0.27778
        # Marginal A = (0.300 - 0.27778) * 500 ≈ 11.111
        # Marginal B = (0.250 - 0.27778) * 400 ≈ -11.111
        stats = [
            {"h": 150.0, "ab": 500.0},
            {"h": 100.0, "ab": 400.0},
        ]
        categories = [_rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories)
        assert result[0]["avg"] == pytest.approx(11.111, abs=0.01)
        assert result[1]["avg"] == pytest.approx(-11.111, abs=0.01)

    def test_rate_stat_lower_era(self) -> None:
        # ERA lower is better, so invert: (baseline - player_rate) * volume
        # Player A: ERA 3.0 in 200 IP, Player B: ERA 4.0 in 100 IP
        # Baseline = (600+400)/(200+100) = 1000/300 ≈ 3.3333
        # Marginal A = (3.3333 - 3.0) * 200 ≈ 66.667
        # Marginal B = (3.3333 - 4.0) * 100 ≈ -66.667
        stats = [
            {"er": 600.0, "ip": 200.0},
            {"er": 400.0, "ip": 100.0},
        ]
        categories = [_rate("era", "er", "ip", Direction.LOWER)]
        result = convert_rate_stats(stats, categories)
        assert result[0]["era"] == pytest.approx(66.667, abs=0.01)
        assert result[1]["era"] == pytest.approx(-66.667, abs=0.01)

    def test_rate_stat_compound_numerator_whip(self) -> None:
        # WHIP = (bb+h)/ip, lower is better
        # Player A: (50+150)/200 = 1.0, Player B: (40+120)/100 = 1.6
        # Baseline = (50+150+40+120)/(200+100) = 360/300 = 1.2
        # Marginal A = (1.2 - 1.0) * 200 = 40.0
        # Marginal B = (1.2 - 1.6) * 100 = -40.0
        stats = [
            {"bb": 50.0, "h": 150.0, "ip": 200.0},
            {"bb": 40.0, "h": 120.0, "ip": 100.0},
        ]
        categories = [_rate("whip", "bb+h", "ip", Direction.LOWER)]
        result = convert_rate_stats(stats, categories)
        assert result[0]["whip"] == pytest.approx(40.0, abs=0.01)
        assert result[1]["whip"] == pytest.approx(-40.0, abs=0.01)

    def test_mixed_categories(self) -> None:
        stats = [
            {"hr": 30.0, "h": 150.0, "ab": 500.0},
            {"hr": 20.0, "h": 100.0, "ab": 400.0},
        ]
        categories = [_counting("hr"), _rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories)
        # Counting preserved
        assert result[0]["hr"] == 30.0
        # Rate converted
        assert "avg" in result[0]
        assert "avg" in result[1]

    def test_missing_numerator_gives_zero_marginal(self) -> None:
        stats = [{"ab": 500.0}, {"h": 100.0, "ab": 400.0}]
        categories = [_rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories)
        # Player 0 missing "h" → numerator=0, rate=0/500=0
        # Baseline = (0+100)/(500+400) ≈ 0.1111
        # Marginal = (0 - 0.1111)*500 ≈ -55.556
        assert result[0]["avg"] == pytest.approx(-55.556, abs=0.01)

    def test_denominator_zero_gives_zero_marginal(self) -> None:
        stats = [{"h": 0.0, "ab": 0.0}, {"h": 100.0, "ab": 400.0}]
        categories = [_rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories)
        assert result[0]["avg"] == 0.0

    def test_empty_pool(self) -> None:
        categories = [_rate("avg", "h", "ab")]
        result = convert_rate_stats([], categories)
        assert result == []

    def test_single_player_marginal_zero(self) -> None:
        # Single player: baseline = player rate → marginal = 0
        stats = [{"h": 150.0, "ab": 500.0}]
        categories = [_rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories)
        assert result[0]["avg"] == pytest.approx(0.0)

    def test_counting_lower_negated(self) -> None:
        # Counting + Direction.LOWER → negate the value
        stats = [{"er": 60.0}, {"er": 40.0}]
        categories = [_counting("er", Direction.LOWER)]
        result = convert_rate_stats(stats, categories)
        assert result[0]["er"] == -60.0
        assert result[1]["er"] == -40.0

    def test_counting_composite_key_sums_components(self) -> None:
        stats = [{"sv": 30.0, "hld": 20.0}, {"sv": 10.0, "hld": 15.0}]
        categories = [_counting("sv+hld")]
        result = convert_rate_stats(stats, categories)
        assert result[0]["sv+hld"] == 50.0
        assert result[1]["sv+hld"] == 25.0

    def test_counting_composite_missing_component_defaults_zero(self) -> None:
        stats = [{"sv": 30.0}, {"sv": 10.0, "hld": 15.0}]
        categories = [_counting("sv+hld")]
        result = convert_rate_stats(stats, categories)
        assert result[0]["sv+hld"] == 30.0
        assert result[1]["sv+hld"] == 25.0

    def test_counting_composite_lower_negated(self) -> None:
        stats = [{"sv": 30.0, "hld": 20.0}]
        categories = [_counting("sv+hld", Direction.LOWER)]
        result = convert_rate_stats(stats, categories)
        assert result[0]["sv+hld"] == -50.0

    def test_does_not_mutate_input(self) -> None:
        original = {"hr": 30.0, "h": 150.0, "ab": 500.0}
        stats = [original.copy()]
        categories = [_counting("hr"), _rate("avg", "h", "ab")]
        convert_rate_stats(stats, categories)
        assert stats[0] == original


class TestComputeZScores:
    def test_two_players_one_category(self) -> None:
        # HR: [30, 20] → mean=25, std=5 → z=[1.0, -1.0]
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        result = compute_z_scores(stats, ["hr"])
        assert result[0].category_z["hr"] == pytest.approx(1.0)
        assert result[1].category_z["hr"] == pytest.approx(-1.0)

    def test_three_players_two_categories(self) -> None:
        stats = [
            {"hr": 30.0, "rbi": 100.0},
            {"hr": 20.0, "rbi": 80.0},
            {"hr": 25.0, "rbi": 90.0},
        ]
        result = compute_z_scores(stats, ["hr", "rbi"])
        assert len(result) == 3
        # Each result should have both categories
        for pz in result:
            assert "hr" in pz.category_z
            assert "rbi" in pz.category_z

    def test_zero_std_dev_gives_zero_z(self) -> None:
        # All same value → std=0 → z=0 for all
        stats = [{"hr": 25.0}, {"hr": 25.0}, {"hr": 25.0}]
        result = compute_z_scores(stats, ["hr"])
        for pz in result:
            assert pz.category_z["hr"] == 0.0

    def test_single_player(self) -> None:
        stats = [{"hr": 30.0}]
        result = compute_z_scores(stats, ["hr"])
        assert len(result) == 1
        # std=0 → z=0
        assert result[0].category_z["hr"] == 0.0
        assert result[0].composite_z == 0.0

    def test_empty_pool(self) -> None:
        result = compute_z_scores([], ["hr"])
        assert result == []

    def test_composite_is_sum_of_category_z(self) -> None:
        stats = [
            {"hr": 30.0, "rbi": 100.0},
            {"hr": 20.0, "rbi": 80.0},
        ]
        result = compute_z_scores(stats, ["hr", "rbi"])
        for pz in result:
            expected = sum(pz.category_z.values())
            assert pz.composite_z == pytest.approx(expected)

    def test_z_scores_have_zero_mean(self) -> None:
        stats = [{"hr": 10.0}, {"hr": 20.0}, {"hr": 30.0}, {"hr": 40.0}]
        result = compute_z_scores(stats, ["hr"])
        mean_z = sum(pz.category_z["hr"] for pz in result) / len(result)
        assert mean_z == pytest.approx(0.0, abs=1e-10)

    def test_z_scores_have_unit_std_dev(self) -> None:
        stats = [{"hr": 10.0}, {"hr": 20.0}, {"hr": 30.0}, {"hr": 40.0}]
        result = compute_z_scores(stats, ["hr"])
        zs = [pz.category_z["hr"] for pz in result]
        mean_z = sum(zs) / len(zs)
        std_z = (sum((z - mean_z) ** 2 for z in zs) / len(zs)) ** 0.5
        assert std_z == pytest.approx(1.0)

    def test_player_index_matches_position(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        result = compute_z_scores(stats, ["hr"])
        assert result[0].player_index == 0
        assert result[1].player_index == 1

    def test_stdev_overrides_changes_z_scores(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        without = compute_z_scores(stats, ["hr"])
        # Pool stdev = 5.0; override with 10.0 → z-scores halved
        with_override = compute_z_scores(stats, ["hr"], stdev_overrides={"hr": 10.0})
        assert with_override[0].category_z["hr"] == pytest.approx(0.5)
        assert with_override[1].category_z["hr"] == pytest.approx(-0.5)
        assert with_override[0].category_z["hr"] != without[0].category_z["hr"]

    def test_partial_overrides_falls_back(self) -> None:
        stats = [{"hr": 30.0, "rbi": 100.0}, {"hr": 20.0, "rbi": 80.0}]
        # Override hr stdev only; rbi should use pool stdev
        result = compute_z_scores(stats, ["hr", "rbi"], stdev_overrides={"hr": 10.0})
        # hr: mean=25, stdev override=10 → z = (30-25)/10 = 0.5
        assert result[0].category_z["hr"] == pytest.approx(0.5)
        # rbi: mean=90, pool stdev=10 → z = (100-90)/10 = 1.0
        assert result[0].category_z["rbi"] == pytest.approx(1.0)

    def test_no_overrides_unchanged(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        without = compute_z_scores(stats, ["hr"])
        with_none = compute_z_scores(stats, ["hr"], stdev_overrides=None)
        assert without[0].category_z == with_none[0].category_z
        assert without[1].category_z == with_none[1].category_z


class TestCategoryWeights:
    def test_category_weights_applied(self) -> None:
        """Weight of 0.5 on one category halves its contribution to composite."""
        stats = [{"hr": 30.0, "rbi": 100.0}, {"hr": 20.0, "rbi": 80.0}]
        unweighted = compute_z_scores(stats, ["hr", "rbi"])
        weighted = compute_z_scores(stats, ["hr", "rbi"], category_weights={"hr": 0.5, "rbi": 1.0})
        # HR z-scores are the same, but composite differs because HR contribution is halved
        assert weighted[0].category_z["hr"] == unweighted[0].category_z["hr"]
        expected_composite = unweighted[0].category_z["hr"] * 0.5 + unweighted[0].category_z["rbi"]
        assert weighted[0].composite_z == pytest.approx(expected_composite)

    def test_category_weight_zero_removes_from_composite(self) -> None:
        """Weight=0 for a category → same composite as omitting that category entirely."""
        stats = [{"hr": 30.0, "sv": 40.0}, {"hr": 20.0, "sv": 10.0}]
        with_zero = compute_z_scores(stats, ["hr", "sv"], category_weights={"sv": 0.0})
        without_cat = compute_z_scores(stats, ["hr"])
        assert with_zero[0].composite_z == pytest.approx(without_cat[0].composite_z)
        assert with_zero[1].composite_z == pytest.approx(without_cat[1].composite_z)

    def test_category_weights_none_is_backward_compatible(self) -> None:
        """Default (None) → same as all weights=1.0."""
        stats = [{"hr": 30.0, "rbi": 100.0}, {"hr": 20.0, "rbi": 80.0}]
        without = compute_z_scores(stats, ["hr", "rbi"])
        with_none = compute_z_scores(stats, ["hr", "rbi"], category_weights=None)
        with_ones = compute_z_scores(stats, ["hr", "rbi"], category_weights={"hr": 1.0, "rbi": 1.0})
        assert without[0].composite_z == pytest.approx(with_none[0].composite_z)
        assert without[0].composite_z == pytest.approx(with_ones[0].composite_z)

    def test_pipeline_with_category_weights(self) -> None:
        """run_zar_pipeline with weights produces valid results with different z-scores."""
        stats = [{"hr": 30.0, "sv": 40.0}, {"hr": 20.0, "sv": 10.0}]
        categories = [_counting("hr"), _counting("sv")]
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        without = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        with_weights = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=100.0, category_weights={"sv": 0.0}
        )
        # With sv zeroed out, composite z-scores should differ
        without_z = [pz.composite_z for pz in without.z_scores]
        with_z = [pz.composite_z for pz in with_weights.z_scores]
        assert without_z != with_z
        # Composite with weight=0 should equal hr-only z-scores
        hr_only = run_zar_pipeline(stats, [_counting("hr")], positions, roster_spots, num_teams=1, budget=100.0)
        hr_only_z = [pz.composite_z for pz in hr_only.z_scores]
        assert with_z == pytest.approx(hr_only_z)


def _pz(index: int, composite: float) -> PlayerZScores:
    """Shorthand for creating a PlayerZScores with just composite_z."""
    return PlayerZScores(player_index=index, category_z={}, composite_z=composite)


class TestComputeReplacementLevel:
    def test_single_position(self) -> None:
        # 5 players, 2 roster spots, 2 teams → draftable=4, replacement=player at index 4
        z_scores = [_pz(i, 10.0 - i) for i in range(5)]  # 10, 9, 8, 7, 6
        positions = [["C"]] * 5
        roster_spots = {"C": 2}
        result = compute_replacement_level(z_scores, positions, roster_spots, num_teams=2)
        # draftable_count=4, replacement = z at index 4 = 6.0
        assert result["C"] == 6.0

    def test_multiple_positions(self) -> None:
        z_scores = [_pz(i, 10.0 - i) for i in range(6)]
        positions = [["1B"]] * 3 + [["OF"]] * 3
        roster_spots = {"1B": 1, "OF": 1}
        result = compute_replacement_level(z_scores, positions, roster_spots, num_teams=2)
        # 1B: players [0,1,2] sorted desc by composite → [10,9,8], draftable=2, replacement=8.0
        assert result["1B"] == 8.0
        # OF: players [3,4,5] sorted desc → [7,6,5], draftable=2, replacement=5.0
        assert result["OF"] == 5.0

    def test_multi_position_eligibility(self) -> None:
        # Player eligible at multiple positions appears in each position's pool
        z_scores = [_pz(0, 10.0), _pz(1, 8.0), _pz(2, 6.0)]
        positions = [["1B", "OF"], ["1B"], ["OF"]]
        roster_spots = {"1B": 1, "OF": 1}
        result = compute_replacement_level(z_scores, positions, roster_spots, num_teams=1)
        # 1B pool: [10, 8], draftable=1, replacement=8.0
        assert result["1B"] == 8.0
        # OF pool: [10, 6], draftable=1, replacement=6.0
        assert result["OF"] == 6.0

    def test_fewer_players_than_slots(self) -> None:
        # Only 2 players but 6 draftable slots
        z_scores = [_pz(0, 10.0), _pz(1, 5.0)]
        positions = [["C"], ["C"]]
        roster_spots = {"C": 3}
        result = compute_replacement_level(z_scores, positions, roster_spots, num_teams=2)
        # draftable=6 > 2 players → use minimum = 5.0
        assert result["C"] == 5.0

    def test_empty_input(self) -> None:
        result = compute_replacement_level([], [], {}, num_teams=12)
        assert result == {}

    def test_position_not_in_roster_spots(self) -> None:
        # Player has position not listed in roster_spots → that position not in result
        z_scores = [_pz(0, 10.0)]
        positions = [["DH"]]
        roster_spots = {"C": 1}
        result = compute_replacement_level(z_scores, positions, roster_spots, num_teams=12)
        assert result.get("C") == 0.0
        assert "DH" not in result

    def test_no_players_at_position(self) -> None:
        # Position in roster_spots but no players eligible → replacement = 0.0
        z_scores = [_pz(0, 10.0)]
        positions = [["1B"]]
        roster_spots = {"1B": 1, "C": 1}
        result = compute_replacement_level(z_scores, positions, roster_spots, num_teams=1)
        assert result["C"] == 0.0


class TestComputeVar:
    def test_basic_var(self) -> None:
        z_scores = [_pz(0, 10.0), _pz(1, 5.0)]
        replacement = {"C": 3.0}
        positions = [["C"], ["C"]]
        result = compute_var(z_scores, replacement, positions)
        assert result[0] == pytest.approx(7.0)  # 10 - 3
        assert result[1] == pytest.approx(2.0)  # 5 - 3

    def test_best_position_chosen(self) -> None:
        # Player eligible at 1B (repl=5) and OF (repl=3) → uses OF (lower repl = higher VAR)
        z_scores = [_pz(0, 10.0)]
        replacement = {"1B": 5.0, "OF": 3.0}
        positions = [["1B", "OF"]]
        result = compute_var(z_scores, replacement, positions)
        assert result[0] == pytest.approx(7.0)  # 10 - 3

    def test_negative_var(self) -> None:
        z_scores = [_pz(0, 2.0)]
        replacement = {"C": 5.0}
        positions = [["C"]]
        result = compute_var(z_scores, replacement, positions)
        assert result[0] == pytest.approx(-3.0)

    def test_no_positions_fallback(self) -> None:
        # Player with no recognized positions uses max replacement as baseline
        z_scores = [_pz(0, 10.0)]
        replacement = {"C": 3.0, "1B": 5.0}
        positions = [[]]
        result = compute_var(z_scores, replacement, positions)
        assert result[0] == pytest.approx(5.0)  # 10 - max(3, 5)

    def test_empty_input(self) -> None:
        result = compute_var([], {}, [])
        assert result == []

    def test_single_position_all_players(self) -> None:
        z_scores = [_pz(0, 10.0), _pz(1, 8.0), _pz(2, 6.0)]
        replacement = {"SS": 4.0}
        positions = [["SS"], ["SS"], ["SS"]]
        result = compute_var(z_scores, replacement, positions)
        assert result == [pytest.approx(6.0), pytest.approx(4.0), pytest.approx(2.0)]


class TestVarToDollars:
    def test_proportional_distribution(self) -> None:
        # 2 players with VAR 6 and 4, budget=20, min_bid=1
        # surplus = 20 - 2*1 = 18
        # player 0: 1 + (6/10)*18 = 1 + 10.8 = 11.8
        # player 1: 1 + (4/10)*18 = 1 + 7.2 = 8.2
        result = var_to_dollars([6.0, 4.0], total_budget=20.0)
        assert result[0] == pytest.approx(11.8)
        assert result[1] == pytest.approx(8.2)

    def test_negative_var_gets_min_bid(self) -> None:
        result = var_to_dollars([10.0, -5.0], total_budget=20.0)
        assert result[1] == pytest.approx(1.0)
        # Player 0 gets the surplus: 1 + (10/10)*(20-2) = 1 + 18 = 19.0
        assert result[0] == pytest.approx(19.0)

    def test_total_adds_to_budget(self) -> None:
        var_values = [10.0, 5.0, 3.0, -2.0, 0.0]
        budget = 100.0
        result = var_to_dollars(var_values, total_budget=budget)
        assert sum(result) == pytest.approx(budget)

    def test_all_same_var(self) -> None:
        result = var_to_dollars([5.0, 5.0, 5.0], total_budget=30.0)
        # Each gets equal share: surplus=27, each gets 1 + 27/3 = 10.0
        for d in result:
            assert d == pytest.approx(10.0)

    def test_no_positive_var(self) -> None:
        result = var_to_dollars([-1.0, -2.0, 0.0], total_budget=30.0)
        for d in result:
            assert d == pytest.approx(1.0)

    def test_empty_input(self) -> None:
        result = var_to_dollars([], total_budget=260.0)
        assert result == []

    def test_custom_min_bid(self) -> None:
        # 2 players, budget=20, min_bid=2
        # surplus = 20 - 2*2 = 16
        # player 0: 2 + (6/10)*16 = 2 + 9.6 = 11.6
        # player 1: 2 + (4/10)*16 = 2 + 6.4 = 8.4
        result = var_to_dollars([6.0, 4.0], total_budget=20.0, min_bid=2.0)
        assert result[0] == pytest.approx(11.6)
        assert result[1] == pytest.approx(8.4)

    def test_roster_spots_total_limits_draftable(self) -> None:
        # 4 players with VAR [10, 5, 2, -1], roster_spots_total=2, budget=100
        # Top 2 positive-VAR players are draftable (VAR 10 and 5)
        # n_draftable=2, surplus = 100 - 2*1 = 98, sum_draftable=15
        # Player 0: 1 + (10/15)*98 = 1 + 65.333 = 66.333
        # Player 1: 1 + (5/15)*98 = 1 + 32.667 = 33.667
        # Players 2, 3: 0.0
        result = var_to_dollars([10.0, 5.0, 2.0, -1.0], total_budget=100.0, roster_spots_total=2)
        assert result[0] == pytest.approx(66.333, abs=0.01)
        assert result[1] == pytest.approx(33.667, abs=0.01)

    def test_roster_spots_total_non_draftable_get_zero(self) -> None:
        result = var_to_dollars([10.0, 5.0, 2.0, -1.0], total_budget=100.0, roster_spots_total=2)
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_roster_spots_total_draftable_sum_to_budget(self) -> None:
        # 5 players, roster_spots_total=3
        var_values = [10.0, 8.0, 5.0, 2.0, -1.0]
        budget = 200.0
        result = var_to_dollars(var_values, total_budget=budget, roster_spots_total=3)
        draftable_sum = sum(d for d in result if d > 0.0)
        assert draftable_sum == pytest.approx(budget)

    def test_roster_spots_total_exceeds_positive_var_players(self) -> None:
        # roster_spots_total=10 but only 3 players have positive VAR
        var_values = [10.0, 5.0, 2.0, 0.0, -1.0, -3.0]
        result = var_to_dollars(var_values, total_budget=100.0, roster_spots_total=10)
        # All 3 positive-VAR players are draftable; the rest get $0
        assert result[0] > 0.0
        assert result[1] > 0.0
        assert result[2] > 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0
        assert result[5] == 0.0

    def test_roster_spots_total_preserves_input_order(self) -> None:
        # Input not sorted by VAR; output must match original input order
        var_values = [2.0, 10.0, -1.0, 5.0]
        result = var_to_dollars(var_values, total_budget=100.0, roster_spots_total=2)
        # Top 2 by VAR: index 1 (10.0) and index 3 (5.0) are draftable
        # n_draftable=2, surplus = 100 - 2 = 98, sum_draftable=15
        assert result[0] == 0.0  # VAR=2.0, not in top 2
        assert result[1] == pytest.approx(1.0 + (10.0 / 15.0) * 98.0)  # draftable
        assert result[2] == 0.0  # VAR=-1.0
        assert result[3] == pytest.approx(1.0 + (5.0 / 15.0) * 98.0)  # draftable

    def test_roster_spots_total_none_preserves_current_behavior(self) -> None:
        # Explicitly pass roster_spots_total=None, same result as before
        result_none = var_to_dollars([6.0, 4.0], total_budget=20.0, roster_spots_total=None)
        result_default = var_to_dollars([6.0, 4.0], total_budget=20.0)
        assert result_none == result_default


def _league(
    batting_categories: tuple[CategoryConfig, ...] = (),
    pitching_categories: tuple[CategoryConfig, ...] = (),
    teams: int = 2,
    budget: int = 260,
    roster_batters: int = 3,
    roster_pitchers: int = 2,
    budget_split: BudgetSplitMode = BudgetSplitMode.ROSTER_SPOTS,
    budget_hitter_pct: float | None = None,
) -> LeagueSettings:
    return LeagueSettings(
        name="Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=teams,
        budget=budget,
        roster_batters=roster_batters,
        roster_pitchers=roster_pitchers,
        batting_categories=batting_categories,
        pitching_categories=pitching_categories,
        budget_split=budget_split,
        budget_hitter_pct=budget_hitter_pct,
    )


class TestComputeBudgetSplit:
    def test_default_uses_roster_spots(self) -> None:
        """Default mode (ROSTER_SPOTS) splits by roster_batters / roster_pitchers."""
        league = _league(
            batting_categories=(_counting("hr"), _counting("r")),
            pitching_categories=(_counting("w"), _counting("sv")),
            teams=1,
            budget=100,
            roster_batters=3,
            roster_pitchers=2,
        )
        bat_budget, pitch_budget = compute_budget_split(league)
        # 3/(3+2) = 60%, 2/(3+2) = 40%
        assert bat_budget == pytest.approx(60.0)
        assert pitch_budget == pytest.approx(40.0)

    def test_categories_mode_explicit(self) -> None:
        """Explicit CATEGORIES mode uses category count ratio."""
        league = _league(
            batting_categories=(_counting("hr"), _counting("r"), _counting("rbi")),
            pitching_categories=(_counting("w"),),
            teams=2,
            budget=260,
            budget_split=BudgetSplitMode.CATEGORIES,
        )
        bat_budget, pitch_budget = compute_budget_split(league)
        total = 260 * 2
        assert bat_budget == pytest.approx(total * 3 / 4)
        assert pitch_budget == pytest.approx(total * 1 / 4)

    def test_zero_roster_totals(self) -> None:
        """roster_batters=0 and roster_pitchers=0 returns (0.0, 0.0)."""
        league = _league(roster_batters=0, roster_pitchers=0)
        bat_budget, pitch_budget = compute_budget_split(league)
        assert bat_budget == 0.0
        assert pitch_budget == 0.0

    def test_zero_total_categories(self) -> None:
        """CATEGORIES mode with no categories returns (0.0, 0.0)."""
        league = _league(budget_split=BudgetSplitMode.CATEGORIES)
        bat_budget, pitch_budget = compute_budget_split(league)
        assert bat_budget == 0.0
        assert pitch_budget == 0.0

    def test_roster_spots_proportional(self) -> None:
        """H2H-like config: 9 batters, 8 pitchers, $260x12."""
        league = _league(
            batting_categories=(_counting("hr"),) * 5,
            pitching_categories=(_counting("w"),) * 5,
            teams=12,
            budget=260,
            roster_batters=9,
            roster_pitchers=8,
        )
        bat_budget, pitch_budget = compute_budget_split(league)
        total = 260 * 12  # 3120
        assert bat_budget == pytest.approx(total * 9 / 17)  # ~1651.76
        assert pitch_budget == pytest.approx(total * 8 / 17)  # ~1468.24
        assert bat_budget + pitch_budget == pytest.approx(total)

    def test_fixed_ratio_67_33(self) -> None:
        """FIXED_RATIO with 67% hitter split."""
        league = _league(
            teams=1,
            budget=100,
            budget_split=BudgetSplitMode.FIXED_RATIO,
            budget_hitter_pct=0.67,
        )
        bat_budget, pitch_budget = compute_budget_split(league)
        assert bat_budget == pytest.approx(67.0)
        assert pitch_budget == pytest.approx(33.0)

    def test_fixed_ratio_zero(self) -> None:
        """FIXED_RATIO with 0% hitter → all budget to pitchers."""
        league = _league(
            teams=1,
            budget=100,
            budget_split=BudgetSplitMode.FIXED_RATIO,
            budget_hitter_pct=0.0,
        )
        bat_budget, pitch_budget = compute_budget_split(league)
        assert bat_budget == pytest.approx(0.0)
        assert pitch_budget == pytest.approx(100.0)

    def test_fixed_ratio_one(self) -> None:
        """FIXED_RATIO with 100% hitter → all budget to batters."""
        league = _league(
            teams=1,
            budget=100,
            budget_split=BudgetSplitMode.FIXED_RATIO,
            budget_hitter_pct=1.0,
        )
        bat_budget, pitch_budget = compute_budget_split(league)
        assert bat_budget == pytest.approx(100.0)
        assert pitch_budget == pytest.approx(0.0)

    def test_fixed_ratio_half(self) -> None:
        """FIXED_RATIO with 50% split."""
        league = _league(
            teams=2,
            budget=260,
            budget_split=BudgetSplitMode.FIXED_RATIO,
            budget_hitter_pct=0.5,
        )
        bat_budget, pitch_budget = compute_budget_split(league)
        assert bat_budget == pytest.approx(260.0)
        assert pitch_budget == pytest.approx(260.0)

    def test_fixed_ratio_missing_pct_raises(self) -> None:
        """FIXED_RATIO without budget_hitter_pct raises ValueError."""
        league = _league(budget_split=BudgetSplitMode.FIXED_RATIO)
        with pytest.raises(ValueError, match="budget_hitter_pct"):
            compute_budget_split(league)


class TestRunZarPipeline:
    def test_returns_pipeline_result(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        result = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        assert isinstance(result, ZarPipelineResult)

    def test_empty_input(self) -> None:
        result = run_zar_pipeline([], [_counting("hr")], [], {}, num_teams=1, budget=100.0)
        assert result.z_scores == []
        assert result.replacement == {}
        assert result.dollar_values == []

    def test_dollar_values_sum_to_budget(self) -> None:
        # 3 players all at "of", 3 roster spots, 1 team → all draftable
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 3}
        result = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=90.0)
        draftable_sum = sum(d for d in result.dollar_values if d > 0.0)
        assert draftable_sum == pytest.approx(90.0)

    def test_z_scores_length_matches_input(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        result = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        assert len(result.z_scores) == 2
        assert len(result.dollar_values) == 2

    def test_replacement_keys_match_roster_spots(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        positions = [["c", "of"], ["of"]]
        roster_spots = {"c": 1, "of": 1}
        result = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        assert set(result.replacement.keys()) == {"c", "of"}

    def test_stdev_overrides_passed_through(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 3}
        without = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=90.0)
        with_override = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=90.0, stdev_overrides={"hr": 100.0}
        )
        # Different stdev → different z-scores
        without_z = [pz.composite_z for pz in without.z_scores]
        override_z = [pz.composite_z for pz in with_override.z_scores]
        assert without_z != override_z

    def test_known_value_regression(self) -> None:
        # 2 players, 1 counting category (hr)
        # Player A: hr=30, Player B: hr=20
        # After convert_rate_stats: [30.0, 20.0] (counting, higher)
        # z-scores: mean=25, pstdev=5 → z_A=1.0, z_B=-1.0
        # Replacement for "of" with 1 spot, 1 team: draftable=1, repl = z at index 1 = -1.0
        # VAR: A=1.0-(-1.0)=2.0, B=-1.0-(-1.0)=0.0
        # var_to_dollars (draftable-only): roster_spots_total=1, budget=20
        # Only Player A has positive VAR (2.0), Player B has VAR=0.0 → non-draftable
        # n_draftable=1, surplus=20-1=19, A: 1 + (2.0/2.0)*19 = 20.0, B: 0.0
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        result = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=20.0, use_optimal_assignment=False
        )
        assert result.z_scores[0].composite_z == pytest.approx(1.0)
        assert result.z_scores[1].composite_z == pytest.approx(-1.0)
        assert result.replacement["of"] == pytest.approx(-1.0)
        assert result.dollar_values[0] == pytest.approx(20.0)
        assert result.dollar_values[1] == pytest.approx(0.0)
        assert result.assignments is None


class TestConvertRateStatsDirectRates:
    def test_direct_rate_used_when_present(self) -> None:
        """When stats dict has an 'avg' field that disagrees with h/ab, direct rate wins."""
        # h/ab = 150/500 = 0.300, but avg field says 0.320
        stats = [
            {"h": 150.0, "ab": 500.0, "avg": 0.320},
            {"h": 100.0, "ab": 400.0, "avg": 0.260},
        ]
        categories = [_rate("avg", "h", "ab")]
        result_direct = convert_rate_stats(stats, categories, use_direct_rates=True)
        result_derived = convert_rate_stats(stats, categories, use_direct_rates=False)
        # Direct and derived should differ because avg != h/ab
        assert result_direct[0]["avg"] != pytest.approx(result_derived[0]["avg"])

    def test_baseline_volume_weighted(self) -> None:
        """Baseline should be volume-weighted mean of direct rates."""
        # Player A: avg=0.300, ab=600; Player B: avg=0.250, ab=400
        # Weighted baseline = (0.300*600 + 0.250*400) / (600+400) = (180+100)/1000 = 0.280
        # Marginal A = (0.300 - 0.280) * 600 = 12.0
        # Marginal B = (0.250 - 0.280) * 400 = -12.0
        stats = [
            {"h": 999.0, "ab": 600.0, "avg": 0.300},  # h is irrelevant when direct
            {"h": 999.0, "ab": 400.0, "avg": 0.250},
        ]
        categories = [_rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories, use_direct_rates=True)
        assert result[0]["avg"] == pytest.approx(12.0)
        assert result[1]["avg"] == pytest.approx(-12.0)

    def test_fallback_when_key_missing(self) -> None:
        """Player without direct rate key falls back to derived calculation."""
        # Player A has avg field, Player B does not
        # Player A: avg=0.320, ab=500
        # Player B: h=100, ab=400 → derived rate = 0.250
        # Baseline: weighted mean = (0.320*500 + 0.250*400)/(500+400) = (160+100)/900 ≈ 0.28889
        # Marginal A = (0.320 - 0.28889) * 500 ≈ 15.556
        # Marginal B = (0.250 - 0.28889) * 400 ≈ -15.556
        stats = [
            {"h": 150.0, "ab": 500.0, "avg": 0.320},
            {"h": 100.0, "ab": 400.0},  # no avg key
        ]
        categories = [_rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories, use_direct_rates=True)
        assert result[0]["avg"] == pytest.approx(15.556, abs=0.01)
        assert result[1]["avg"] == pytest.approx(-15.556, abs=0.01)

    def test_counting_stats_unaffected(self) -> None:
        """Counting stats produce the same result regardless of the flag."""
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        result_off = convert_rate_stats(stats, categories, use_direct_rates=False)
        result_on = convert_rate_stats(stats, categories, use_direct_rates=True)
        assert result_off == result_on

    def test_mixed_categories(self) -> None:
        """One counting + one rate category with flag on."""
        stats = [
            {"hr": 30.0, "h": 150.0, "ab": 500.0, "avg": 0.320},
            {"hr": 20.0, "h": 100.0, "ab": 400.0, "avg": 0.260},
        ]
        categories = [_counting("hr"), _rate("avg", "h", "ab")]
        result = convert_rate_stats(stats, categories, use_direct_rates=True)
        # Counting: unchanged
        assert result[0]["hr"] == 30.0
        assert result[1]["hr"] == 20.0
        # Rate: uses direct rates (0.320 and 0.260), not h/ab
        # Baseline = (0.320*500 + 0.260*400) / 900 = (160+104)/900 ≈ 0.29333
        assert result[0]["avg"] == pytest.approx((0.320 - 0.29333) * 500, abs=0.1)

    def test_false_preserves_behavior(self) -> None:
        """Explicit use_direct_rates=False produces identical results to no-flag call."""
        stats = [
            {"h": 150.0, "ab": 500.0, "avg": 0.320},
            {"h": 100.0, "ab": 400.0, "avg": 0.260},
        ]
        categories = [_rate("avg", "h", "ab")]
        result_default = convert_rate_stats(stats, categories)
        result_false = convert_rate_stats(stats, categories, use_direct_rates=False)
        assert result_default == result_false

    def test_pipeline_with_direct_rates(self) -> None:
        """run_zar_pipeline with use_direct_rates=True produces different z-scores."""
        # Three players with rate stats that diverge from counting components.
        # With 3 players, z-score normalization doesn't collapse to ±1.
        stats = [
            {"h": 150.0, "ab": 500.0, "avg": 0.350},  # derived=0.300, direct=0.350
            {"h": 100.0, "ab": 400.0, "avg": 0.200},  # derived=0.250, direct=0.200
            {"h": 120.0, "ab": 400.0, "avg": 0.280},  # derived=0.300, direct=0.280
        ]
        categories = [_rate("avg", "h", "ab")]
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 3}
        result_off = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=100.0, use_direct_rates=False
        )
        result_on = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=100.0, use_direct_rates=True
        )
        # Category z-scores should differ because marginal contributions change
        z_off = [pz.category_z["avg"] for pz in result_off.z_scores]
        z_on = [pz.category_z["avg"] for pz in result_on.z_scores]
        assert z_off != z_on


class TestAssignmentToDollars:
    def test_basic_proportional_distribution(self) -> None:
        # 3 assigned players: VAR [6, 4, 0], budget=20, min_bid=1
        # 3 assigned → base cost=3, surplus=17
        # Player 0: 1 + (6/10)*17 = 1 + 10.2 = 11.2
        # Player 1: 1 + (4/10)*17 = 1 + 6.8 = 7.8
        # Player 2 (VAR=0): 1.0 (min bid)
        var_values = [6.0, 4.0, 0.0]
        assignments = {0: "of", 1: "of", 2: "of"}
        result = assignment_to_dollars(var_values, assignments, total_budget=20.0)
        assert result[0] == pytest.approx(11.2)
        assert result[1] == pytest.approx(7.8)
        assert result[2] == pytest.approx(1.0)

    def test_replacement_level_player_gets_min_bid(self) -> None:
        var_values = [5.0, 0.0]
        assignments = {0: "c", 1: "c"}
        result = assignment_to_dollars(var_values, assignments, total_budget=20.0)
        assert result[1] == pytest.approx(1.0)

    def test_unassigned_player_gets_zero(self) -> None:
        var_values = [5.0, 3.0, 0.0]
        assignments = {0: "c", 1: "of"}  # player 2 not assigned
        result = assignment_to_dollars(var_values, assignments, total_budget=20.0)
        assert result[2] == 0.0

    def test_dollar_sum_equals_budget(self) -> None:
        var_values = [10.0, 5.0, 3.0, 0.0, 0.0]
        assignments = {0: "c", 1: "of", 2: "of", 3: "of", 4: "c"}
        budget = 100.0
        result = assignment_to_dollars(var_values, assignments, total_budget=budget)
        assert sum(result) == pytest.approx(budget)

    def test_empty_input(self) -> None:
        result = assignment_to_dollars([], {}, total_budget=100.0)
        assert result == []

    def test_no_assignments(self) -> None:
        result = assignment_to_dollars([5.0, 3.0], {}, total_budget=100.0)
        assert result == [0.0, 0.0]

    def test_all_replacement_level(self) -> None:
        # All assigned players have VAR=0 → distribute budget evenly
        var_values = [0.0, 0.0, 0.0]
        assignments = {0: "c", 1: "of", 2: "of"}
        result = assignment_to_dollars(var_values, assignments, total_budget=30.0)
        assert result == [pytest.approx(10.0), pytest.approx(10.0), pytest.approx(10.0)]
        assert sum(result) == pytest.approx(30.0)


class TestRunOptimalPipeline:
    def test_dollar_values_sum_to_budget(self) -> None:
        scores = [10.0, 8.0, 5.0, 3.0, 1.0]
        positions = [["c", "of"]] * 5
        roster_spots = {"c": 1, "of": 1}
        budget = 100.0
        replacement, dollars, assignments = run_optimal_pipeline(scores, positions, roster_spots, 1, budget)
        assert sum(dollars) == pytest.approx(budget)

    def test_assigned_count_equals_total_slots(self) -> None:
        scores = [10.0, 8.0, 5.0, 3.0, 1.0]
        positions = [["of"]] * 5
        roster_spots = {"of": 2}
        _, _, assignments = run_optimal_pipeline(scores, positions, roster_spots, 1, 100.0)
        assert len(assignments) == 2

    def test_replacement_derived_from_worst_assigned(self) -> None:
        scores = [10.0, 5.0, 1.0]
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 2}
        replacement, _, assignments = run_optimal_pipeline(scores, positions, roster_spots, 1, 100.0)
        # 2 slots, top 2 assigned (scores 10.0 and 5.0), replacement = 5.0
        assert replacement["of"] == pytest.approx(5.0)

    def test_empty_input(self) -> None:
        replacement, dollars, assignments = run_optimal_pipeline([], [], {}, 1, 100.0)
        assert replacement == {}
        assert dollars == []
        assert assignments == {}


class TestRunZarPipelineOptimal:
    def test_optimal_assignment_default(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 2}
        result = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        assert result.assignments is not None
        assert len(result.assignments) == 2

    def test_optimal_dollar_sum(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"], ["of"]]
        roster_spots = {"of": 3}
        result = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=90.0)
        assert sum(result.dollar_values) == pytest.approx(90.0)

    def test_greedy_flag_disables_assignments(self) -> None:
        stats = [{"hr": 30.0}, {"hr": 20.0}]
        categories = [_counting("hr")]
        positions = [["of"], ["of"]]
        roster_spots = {"of": 1}
        result = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=100.0, use_optimal_assignment=False
        )
        assert result.assignments is None


class TestNormalizeCompositeZ:
    def test_stdev_matches_reference(self) -> None:
        """After normalization, composite stdev matches reference within tolerance."""
        z_scores = [
            PlayerZScores(player_index=0, category_z={"hr": 1.0}, composite_z=10.0),
            PlayerZScores(player_index=1, category_z={"hr": 0.5}, composite_z=5.0),
            PlayerZScores(player_index=2, category_z={"hr": -0.5}, composite_z=-2.0),
            PlayerZScores(player_index=3, category_z={"hr": -1.0}, composite_z=-8.0),
        ]
        reference_stdev = 3.0
        result = normalize_composite_z(z_scores, reference_stdev)
        composites = [pz.composite_z for pz in result]
        assert statistics.pstdev(composites) == pytest.approx(reference_stdev, rel=0.01)

    def test_category_z_unchanged(self) -> None:
        """Normalization does not alter category_z values."""
        z_scores = [
            PlayerZScores(player_index=0, category_z={"hr": 1.5, "r": 0.8}, composite_z=10.0),
            PlayerZScores(player_index=1, category_z={"hr": -0.5, "r": 1.2}, composite_z=5.0),
        ]
        result = normalize_composite_z(z_scores, reference_stdev=2.0)
        for original, normalized in zip(z_scores, result, strict=True):
            assert normalized.category_z == original.category_z

    def test_single_player_returned_unchanged(self) -> None:
        """Single-player input is returned unchanged (can't compute stdev)."""
        z_scores = [PlayerZScores(player_index=0, category_z={"hr": 1.0}, composite_z=5.0)]
        result = normalize_composite_z(z_scores, reference_stdev=3.0)
        assert result[0].composite_z == 5.0

    def test_all_equal_composites_returned_unchanged(self) -> None:
        """When pool stdev is 0 (all equal), input is returned unchanged."""
        z_scores = [
            PlayerZScores(player_index=0, category_z={"hr": 1.0}, composite_z=5.0),
            PlayerZScores(player_index=1, category_z={"hr": 0.5}, composite_z=5.0),
        ]
        result = normalize_composite_z(z_scores, reference_stdev=3.0)
        assert result[0].composite_z == 5.0
        assert result[1].composite_z == 5.0

    def test_player_index_preserved(self) -> None:
        """Player indices are preserved through normalization."""
        z_scores = [
            PlayerZScores(player_index=7, category_z={"hr": 1.0}, composite_z=10.0),
            PlayerZScores(player_index=3, category_z={"hr": -1.0}, composite_z=-10.0),
        ]
        result = normalize_composite_z(z_scores, reference_stdev=5.0)
        assert result[0].player_index == 7
        assert result[1].player_index == 3


class TestRunZarPipelineNormalization:
    def test_reference_composite_stdev_scales_composites(self) -> None:
        """Pipeline with reference_composite_stdev rescales composite z-scores."""
        stats = [{"hr": 40.0}, {"hr": 20.0}, {"hr": 10.0}, {"hr": 5.0}]
        categories = [_counting("hr")]
        positions = [["of"]] * 4
        roster_spots = {"of": 3}

        result_without = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        result_with = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=100.0, reference_composite_stdev=5.0
        )
        composites_without = [pz.composite_z for pz in result_without.z_scores]
        composites_with = [pz.composite_z for pz in result_with.z_scores]
        assert composites_without != composites_with
        # Normalized composites should have stdev ~5.0
        assert statistics.pstdev(composites_with) == pytest.approx(5.0, rel=0.01)

    def test_without_reference_stdev_unchanged(self) -> None:
        """Pipeline without reference_composite_stdev is backward compatible."""
        stats = [{"hr": 30.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        positions = [["of"]] * 3
        roster_spots = {"of": 2}

        result_a = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        result_b = run_zar_pipeline(stats, categories, positions, roster_spots, num_teams=1, budget=100.0)
        assert result_a.dollar_values == result_b.dollar_values

    def test_dollar_sum_preserved_with_normalization(self) -> None:
        """Budget total is preserved when normalization is applied."""
        stats = [{"hr": 40.0}, {"hr": 20.0}, {"hr": 10.0}]
        categories = [_counting("hr")]
        positions = [["of"]] * 3
        roster_spots = {"of": 3}
        budget = 90.0

        result = run_zar_pipeline(
            stats, categories, positions, roster_spots, num_teams=1, budget=budget, reference_composite_stdev=2.0
        )
        assert sum(result.dollar_values) == pytest.approx(budget)
