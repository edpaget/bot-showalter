from statistics import mean

import pytest

from fantasy_baseball_manager.domain.league_settings import (
    LeagueFormat,
    LeagueSettings,
)
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.positional_scarcity import compute_scarcity


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
