import pytest

from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    StatType,
)
from fantasy_baseball_manager.models.zar.engine import (
    PlayerZScores,
    compute_replacement_level,
    compute_var,
    compute_z_scores,
    convert_rate_stats,
    resolve_numerator,
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
        assert "C" == pytest.approx(0.0) or result.get("C") == 0.0
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
