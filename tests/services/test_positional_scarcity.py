from statistics import mean

import pytest

from fantasy_baseball_manager.domain.league_settings import (
    LeagueFormat,
    LeagueSettings,
)
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.positional_scarcity import (
    compute_scarcity,
    compute_scarcity_rankings,
    compute_value_curves,
    scarcity_adjusted_value,
)


def _make_valuation(
    position: str,
    value: float,
    player_id: int = 1,
    season: int = 2026,
) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=season,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="1.0",
        player_type="batter",
        position=position,
        value=value,
        rank=1,
        category_scores={},
    )


def _make_league(
    positions: dict[str, int] | None = None,
    pitcher_positions: dict[str, int] | None = None,
    teams: int = 12,
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=teams,
        budget=260,
        roster_batters=14,
        roster_pitchers=10,
        batting_categories=(),
        pitching_categories=(),
        positions=positions or {"ss": 1},
        pitcher_positions=pitcher_positions or {},
    )


class TestTier1ValueCalculation:
    def test_tier_1_value_calculation(self) -> None:
        """12 players at SS, league has 12 teams × 1 SS slot → N=12.
        tier_1_value = mean of top 12 values."""
        values = [30.0 - i for i in range(15)]  # 30, 29, 28, ..., 16
        valuations = [_make_valuation("ss", v, player_id=i) for i, v in enumerate(values)]
        league = _make_league(positions={"ss": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        assert len(result) == 1
        assert result[0].position == "ss"
        expected = mean(values[:12])  # top 12
        assert result[0].tier_1_value == pytest.approx(expected)


class TestReplacementValue:
    def test_replacement_value(self) -> None:
        """replacement_value = value at rank N+1 (13th player)."""
        values = [30.0 - i for i in range(15)]
        valuations = [_make_valuation("ss", v, player_id=i) for i, v in enumerate(values)]
        league = _make_league(positions={"ss": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        assert result[0].replacement_value == pytest.approx(values[12])  # 18.0


class TestTotalSurplus:
    def test_total_surplus(self) -> None:
        """total_surplus = sum of (value - replacement) for top N starters."""
        values = [30.0 - i for i in range(15)]
        valuations = [_make_valuation("ss", v, player_id=i) for i, v in enumerate(values)]
        league = _make_league(positions={"ss": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        replacement = values[12]  # 18.0
        expected_surplus = sum(v - replacement for v in values[:12])
        assert result[0].total_surplus == pytest.approx(expected_surplus)


class TestDropoffSlope:
    def test_dropoff_slope_steep_vs_flat(self) -> None:
        """Two positions, same tier_1 avg, different slopes.
        Steep position has more negative slope."""
        # SS: steep drop from 30 to 10 over 15 players
        ss_values = [30.0 - i * (20.0 / 14) for i in range(15)]
        # OF: flat, 20 ± small variation over 15 players
        of_values = [20.0 - i * 0.2 for i in range(15)]

        valuations = [_make_valuation("ss", v, player_id=i) for i, v in enumerate(ss_values)]
        valuations += [_make_valuation("of", v, player_id=100 + i) for i, v in enumerate(of_values)]

        league = _make_league(positions={"ss": 1, "of": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        # Results are sorted by slope (most negative first)
        ss_result = next(r for r in result if r.position == "ss")
        of_result = next(r for r in result if r.position == "of")
        assert ss_result.dropoff_slope < of_result.dropoff_slope


class TestElbowDetection:
    def test_elbow_detection(self) -> None:
        """Position with clear cliff: high values then sharp drop.
        steep_rank should be detected near the cliff."""
        # First 6 players: ~25, then sharp drop at rank 7
        values = [25.0, 24.5, 24.0, 23.5, 23.0, 22.5, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
        valuations = [_make_valuation("c", v, player_id=i) for i, v in enumerate(values)]
        league = _make_league(positions={"c": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        assert result[0].steep_rank is not None
        # The cliff is between rank 6 and 7 (0-indexed: 5 and 6)
        assert 5 <= result[0].steep_rank <= 8

    def test_no_elbow_returns_none(self) -> None:
        """Linear decline → steep_rank is None."""
        values = [20.0 - i * 1.0 for i in range(15)]
        valuations = [_make_valuation("ss", v, player_id=i) for i, v in enumerate(values)]
        league = _make_league(positions={"ss": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        assert result[0].steep_rank is None


class TestScarcitySorting:
    def test_scarcity_sorting(self) -> None:
        """Multiple positions, result sorted by steepest slope first."""
        # SS: steep drop
        ss_values = [30.0 - i * 2.0 for i in range(15)]
        # OF: moderate drop
        of_values = [25.0 - i * 1.0 for i in range(15)]
        # C: very steep drop
        c_values = [35.0 - i * 3.0 for i in range(15)]

        valuations = [_make_valuation("ss", v, player_id=i) for i, v in enumerate(ss_values)]
        valuations += [_make_valuation("of", v, player_id=100 + i) for i, v in enumerate(of_values)]
        valuations += [_make_valuation("c", v, player_id=200 + i) for i, v in enumerate(c_values)]

        league = _make_league(positions={"ss": 1, "of": 1, "c": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        assert len(result) == 3
        # Most negative slope first
        slopes = [r.dropoff_slope for r in result]
        assert slopes == sorted(slopes)


class TestEdgeCases:
    def test_empty_valuations(self) -> None:
        """Empty valuations returns empty list."""
        league = _make_league(positions={"ss": 1})
        result = compute_scarcity([], league)
        assert result == []

    def test_position_not_in_league(self) -> None:
        """Valuations at unknown position are excluded."""
        valuations = [_make_valuation("dh", 20.0, player_id=i) for i in range(5)]
        league = _make_league(positions={"ss": 1})

        result = compute_scarcity(valuations, league)

        assert result == []

    def test_position_with_fewer_players_than_slots(self) -> None:
        """Handles gracefully when fewer players than needed slots."""
        # Only 5 catchers when league needs 12
        values = [20.0, 15.0, 10.0, 8.0, 5.0]
        valuations = [_make_valuation("c", v, player_id=i) for i, v in enumerate(values)]
        league = _make_league(positions={"c": 1}, teams=12)

        result = compute_scarcity(valuations, league)

        assert len(result) == 1
        assert result[0].position == "c"
        # tier_1_value is mean of all 5 (since fewer than N=12)
        assert result[0].tier_1_value == pytest.approx(mean(values))
        # replacement_value is last player's value
        assert result[0].replacement_value == pytest.approx(5.0)


class TestComputeValueCurves:
    def test_curve_rank_name_value_tuples(self) -> None:
        """Values are returned as (rank, player_name, value) sorted by value desc."""
        valuations = [
            _make_valuation("ss", 30.0, player_id=1),
            _make_valuation("ss", 20.0, player_id=2),
            _make_valuation("ss", 10.0, player_id=3),
        ]
        league = _make_league(positions={"ss": 1}, teams=2)
        names = {1: "Alice", 2: "Bob", 3: "Carol"}

        curves = compute_value_curves(valuations, league, names)

        assert len(curves) == 1
        curve = curves[0]
        assert curve.position == "ss"
        assert curve.values == [
            (1, "Alice", 30.0),
            (2, "Bob", 20.0),
        ]

    def test_cliff_rank_matches_elbow(self) -> None:
        """cliff_rank is detected at the elbow point."""
        # Clear cliff: steady values then sharp drop
        values = [25.0, 24.5, 24.0, 23.5, 23.0, 22.5, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
        valuations = [_make_valuation("c", v, player_id=i) for i, v in enumerate(values)]
        league = _make_league(positions={"c": 1}, teams=12)
        names = {i: f"Player {i}" for i in range(15)}

        curves = compute_value_curves(valuations, league, names)

        assert len(curves) == 1
        assert curves[0].cliff_rank is not None
        assert 5 <= curves[0].cliff_rank <= 8

    def test_multi_slot_position(self) -> None:
        """OF with 3 slots uses N = teams * 3."""
        valuations = [_make_valuation("of", 20.0 - i, player_id=i) for i in range(10)]
        league = _make_league(positions={"of": 3}, teams=2)
        names = {i: f"Player {i}" for i in range(10)}

        curves = compute_value_curves(valuations, league, names)

        assert len(curves) == 1
        # N = 2 teams * 3 slots = 6
        assert len(curves[0].values) == 6

    def test_missing_player_names_fallback(self) -> None:
        """Players not in the names dict get 'Unknown (id)' as name."""
        valuations = [_make_valuation("ss", 20.0, player_id=99)]
        league = _make_league(positions={"ss": 1}, teams=1)
        names: dict[int, str] = {}  # empty

        curves = compute_value_curves(valuations, league, names)

        assert curves[0].values[0][1] == "Unknown (99)"

    def test_empty_valuations(self) -> None:
        """Empty valuations returns empty list."""
        league = _make_league(positions={"ss": 1})
        result = compute_value_curves([], league, {})
        assert result == []

    def test_position_not_in_league(self) -> None:
        """Valuations at unknown position are excluded."""
        valuations = [_make_valuation("dh", 20.0, player_id=1)]
        league = _make_league(positions={"ss": 1})

        curves = compute_value_curves(valuations, league, {1: "Player"})

        assert curves == []

    def test_multiple_positions(self) -> None:
        """Returns one curve per position that has valuations."""
        valuations = [
            _make_valuation("ss", 30.0, player_id=1),
            _make_valuation("ss", 20.0, player_id=2),
            _make_valuation("c", 25.0, player_id=3),
        ]
        league = _make_league(positions={"ss": 1, "c": 1}, teams=1)
        names = {1: "A", 2: "B", 3: "C"}

        curves = compute_value_curves(valuations, league, names)

        positions = {c.position for c in curves}
        assert positions == {"ss", "c"}


class TestScarcityAdjustedValue:
    def test_zero_score_no_change(self) -> None:
        """A scarcity score of 0 leaves the value unchanged."""
        assert scarcity_adjusted_value(20.0, 0.0) == pytest.approx(20.0)

    def test_full_score_doubles(self) -> None:
        """A scarcity score of 1.0 doubles the value."""
        assert scarcity_adjusted_value(20.0, 1.0) == pytest.approx(40.0)

    def test_partial_proportional(self) -> None:
        """A scarcity score of 0.5 adds 50% to the value."""
        assert scarcity_adjusted_value(20.0, 0.5) == pytest.approx(30.0)

    def test_negative_value(self) -> None:
        """Negative values are also scaled correctly."""
        assert scarcity_adjusted_value(-10.0, 0.5) == pytest.approx(-15.0)


class TestComputeScarcityRankings:
    def _make_scarce_and_deep_valuations(self) -> tuple[list[Valuation], LeagueSettings, dict[int, str]]:
        """Create valuations where C is scarce (steep drop) and OF is deep (flat).

        C: 30, 28, 26, 12, 10, 8 (steep cliff after rank 3)
        OF: 25, 24, 23, 22, 21, 20 (flat)
        """
        c_values = [30.0, 28.0, 26.0, 12.0, 10.0, 8.0]
        of_values = [25.0, 24.0, 23.0, 22.0, 21.0, 20.0]

        valuations: list[Valuation] = []
        names: dict[int, str] = {}
        for i, v in enumerate(c_values):
            valuations.append(_make_valuation("c", v, player_id=100 + i))
            names[100 + i] = f"Catcher {i + 1}"
        for i, v in enumerate(of_values):
            valuations.append(_make_valuation("of", v, player_id=200 + i))
            names[200 + i] = f"Outfielder {i + 1}"

        league = _make_league(positions={"c": 1, "of": 1}, teams=3)
        return valuations, league, names

    def test_scarce_position_boosted(self) -> None:
        """Players at the scarce position (C) get higher adjusted values."""
        valuations, league, names = self._make_scarce_and_deep_valuations()
        result = compute_scarcity_rankings(valuations, league, names)

        # Top C should have adjusted_value > original_value
        top_c = next(p for p in result if p.position == "c")
        assert top_c.adjusted_value > top_c.original_value

    def test_rankings_differ_from_raw(self) -> None:
        """Adjusted rankings differ from raw rankings."""
        valuations, league, names = self._make_scarce_and_deep_valuations()
        result = compute_scarcity_rankings(valuations, league, names)

        original_order = sorted(result, key=lambda p: p.original_rank)
        adjusted_order = sorted(result, key=lambda p: p.adjusted_rank)

        # At least one player should have a different rank
        assert any(o.player_id != a.player_id for o, a in zip(original_order, adjusted_order, strict=True))

    def test_proportional_boost(self) -> None:
        """Players at scarcer positions get larger boosts than those at deep positions."""
        valuations, league, names = self._make_scarce_and_deep_valuations()
        result = compute_scarcity_rankings(valuations, league, names)

        c_players = [p for p in result if p.position == "c"]
        of_players = [p for p in result if p.position == "of"]

        # C scarcity score should be higher than OF
        assert c_players[0].scarcity_score > of_players[0].scarcity_score

    def test_empty_input(self) -> None:
        """Empty valuations return empty list."""
        league = _make_league(positions={"ss": 1})
        result = compute_scarcity_rankings([], league, {})
        assert result == []

    def test_both_values_preserved(self) -> None:
        """Both original and adjusted values are present on results."""
        valuations, league, names = self._make_scarce_and_deep_valuations()
        result = compute_scarcity_rankings(valuations, league, names)

        for p in result:
            assert p.original_value > 0
            assert p.adjusted_value >= p.original_value  # all positive values get boosted

    def test_player_name_fallback(self) -> None:
        """Players not in names dict get 'Unknown (id)' as name."""
        valuations = [_make_valuation("ss", 20.0, player_id=999)]
        league = _make_league(positions={"ss": 1}, teams=1)
        result = compute_scarcity_rankings(valuations, league, {})

        assert result[0].player_name == "Unknown (999)"
