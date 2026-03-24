import pytest

from fantasy_baseball_manager.domain import (
    KeeperConstraints,
    KeeperDecision,
    LeagueFormat,
    LeagueSettings,
    Player,
    Valuation,
)
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.services import (
    compare_scenarios,
    compute_adjusted_draft_pool,
    keeper_trade_impact,
    parse_league_keepers,
    solve_keepers,
    solve_keepers_with_pool,
)


def _decision(
    player_id: int,
    surplus: float,
    position: str = "OF",
    cost: float = 10.0,
    name: str | None = None,
    original_round: int | None = None,
) -> KeeperDecision:
    return KeeperDecision(
        player_id=player_id,
        player_name=name or f"Player {player_id}",
        player_type=PlayerType.BATTER,
        position=position,
        cost=cost,
        projected_value=cost + surplus,
        surplus=surplus,
        years_remaining=1,
        recommendation="keep" if surplus > 0 else "release",
        original_round=original_round,
    )


class TestSolveKeepersBasic:
    def test_selects_highest_surplus(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 10.0),
            _decision(4, 5.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 2}

    def test_keeper_set_fields(self) -> None:
        candidates = [
            _decision(1, 30.0, position="C", cost=15.0),
            _decision(2, 20.0, position="OF", cost=10.0),
            _decision(3, 10.0, position="OF", cost=5.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        assert result.optimal.total_surplus == pytest.approx(50.0)
        assert result.optimal.total_cost == pytest.approx(25.0)
        assert result.optimal.positions_filled == {"C": 1, "OF": 1}
        assert result.optimal.score == pytest.approx(50.0)

    def test_single_candidate(self) -> None:
        candidates = [_decision(1, 15.0)]
        constraints = KeeperConstraints(max_keepers=1)

        result = solve_keepers(candidates, constraints)

        assert len(result.optimal.players) == 1
        assert result.optimal.players[0].player_id == 1

    def test_all_candidates_kept(self) -> None:
        candidates = [_decision(1, 20.0), _decision(2, 10.0)]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        assert len(result.optimal.players) == 2


class TestPositionConstraints:
    def test_position_limit_excludes_excess(self) -> None:
        candidates = [
            _decision(1, 20.0, position="C"),
            _decision(2, 15.0, position="C"),
            _decision(3, 10.0, position="C"),
            _decision(4, 12.0, position="OF"),
        ]
        constraints = KeeperConstraints(max_keepers=2, max_per_position={"C": 1})

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 4}
        assert result.optimal.total_surplus == pytest.approx(32.0)

    def test_multiple_position_limits(self) -> None:
        candidates = [
            _decision(1, 25.0, position="C"),
            _decision(2, 20.0, position="C"),
            _decision(3, 18.0, position="OF"),
            _decision(4, 15.0, position="OF"),
            _decision(5, 12.0, position="OF"),
        ]
        constraints = KeeperConstraints(max_keepers=3, max_per_position={"C": 1, "OF": 2})

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 3, 4}

    def test_unlisted_position_unconstrained(self) -> None:
        candidates = [
            _decision(1, 20.0, position="C"),
            _decision(2, 18.0, position="OF"),
            _decision(3, 15.0, position="OF"),
        ]
        constraints = KeeperConstraints(max_keepers=3, max_per_position={"C": 1})

        result = solve_keepers(candidates, constraints)

        assert len(result.optimal.players) == 3

    def test_position_limit_none(self) -> None:
        candidates = [
            _decision(1, 20.0, position="C"),
            _decision(2, 18.0, position="C"),
            _decision(3, 15.0, position="C"),
        ]
        constraints = KeeperConstraints(max_keepers=3)

        result = solve_keepers(candidates, constraints)

        assert len(result.optimal.players) == 3


class TestCostConstraints:
    def test_max_cost_excludes_expensive(self) -> None:
        candidates = [
            _decision(1, 30.0, cost=20.0),
            _decision(2, 25.0, cost=15.0),
            _decision(3, 20.0, cost=10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2, max_cost=25.0)

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {2, 3}
        assert result.optimal.total_cost <= 25.0

    def test_max_cost_none(self) -> None:
        candidates = [
            _decision(1, 30.0, cost=50.0),
            _decision(2, 25.0, cost=40.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        assert result.optimal.total_cost == pytest.approx(90.0)


class TestRequiredKeepers:
    def test_required_always_included(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 5.0),  # low surplus but required
        ]
        constraints = KeeperConstraints(max_keepers=2, required_keepers=[3])

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert 3 in ids
        assert ids == {1, 3}

    def test_required_with_position_constraint(self) -> None:
        candidates = [
            _decision(1, 25.0, position="C"),
            _decision(2, 20.0, position="C"),  # required
            _decision(3, 15.0, position="OF"),
        ]
        constraints = KeeperConstraints(
            max_keepers=2,
            max_per_position={"C": 1},
            required_keepers=[2],
        )

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        # Player 2 is required (c), so player 1 (c) is blocked by max_per_position
        assert ids == {2, 3}

    def test_required_not_in_candidates_raises(self) -> None:
        candidates = [_decision(1, 20.0)]
        constraints = KeeperConstraints(max_keepers=1, required_keepers=[99])

        with pytest.raises(ValueError, match="required.*99"):
            solve_keepers(candidates, constraints)


class TestAlternatives:
    def test_up_to_5_alternatives(self) -> None:
        # 8 candidates, keep 2 → C(8,2)=28 valid sets, plenty for 5 alternatives
        candidates = [_decision(i, float(20 - i)) for i in range(1, 9)]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        assert len(result.alternatives) <= 5
        assert len(result.alternatives) == 5

    def test_alternatives_distinct_from_optimal(self) -> None:
        candidates = [_decision(i, float(20 - i)) for i in range(1, 9)]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        optimal_ids = frozenset(p.player_id for p in result.optimal.players)
        for alt in result.alternatives:
            alt_ids = frozenset(p.player_id for p in alt.players)
            assert alt_ids != optimal_ids

    def test_fewer_valid_sets(self) -> None:
        # 3 candidates, keep 2 → C(3,2)=3 total sets, so 2 alternatives
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        assert len(result.alternatives) == 2


class TestSensitivity:
    def test_identifies_most_marginal(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 19.0),  # close to player 2
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        # Player 2 is most marginal — replacing with player 3 loses only 1.0
        assert result.sensitivity[0].player_id == 2
        assert result.sensitivity[0].surplus_gap == pytest.approx(1.0)

    def test_sorted_by_gap_ascending(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 19.0),
            _decision(4, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        gaps = [s.surplus_gap for s in result.sensitivity]
        assert gaps == sorted(gaps)


class TestPerformance:
    def test_20_keep_6_under_5s(self) -> None:
        # C(20,6) = 38,760 — enumeration path
        candidates = [_decision(i, float(100 - i * 3)) for i in range(1, 21)]
        constraints = KeeperConstraints(max_keepers=6)

        result = solve_keepers(candidates, constraints)

        # Top 6 by surplus should be players 1-6
        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 2, 3, 4, 5, 6}

    @pytest.mark.slow
    def test_30_keep_8_under_5s(self) -> None:
        # C(30,8) = 5,852,925 — branch-and-bound path
        candidates = [_decision(i, float(200 - i * 4)) for i in range(1, 31)]
        constraints = KeeperConstraints(max_keepers=8)

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 2, 3, 4, 5, 6, 7, 8}


class TestEdgeCases:
    def test_negative_surplus(self) -> None:
        candidates = [
            _decision(1, -5.0),
            _decision(2, -10.0),
            _decision(3, -15.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 2}

    def test_no_valid_sets_raises(self) -> None:
        candidates = [
            _decision(1, 20.0, position="C"),
            _decision(2, 15.0, position="C"),
        ]
        constraints = KeeperConstraints(max_keepers=2, max_per_position={"C": 1})

        with pytest.raises(ValueError, match="no valid"):
            solve_keepers(candidates, constraints)


# ── Round constraint tests ─────────────────────────────────────────────────────


class TestRoundConstraints:
    def test_escalation_bumps_cost(self) -> None:
        """Two players in round 5, escalation=1 bumps both to round 4.
        With max_per_round=1, only one can be kept."""
        candidates = [
            _decision(1, 25.0, original_round=5),
            _decision(2, 20.0, original_round=5),
            _decision(3, 15.0, original_round=3),
        ]
        constraints = KeeperConstraints(
            max_keepers=2,
            round_escalation=1,
            max_per_round=1,
        )

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        # Both escalate from round 5 to round 4, but max_per_round=1 allows only one at round 4.
        # Player 3 escalates from round 3 to round 2, no conflict.
        # Best combo: player 1 (surplus 25) + player 3 (surplus 15) = 40
        assert ids == {1, 3}

    def test_max_per_round_prevents_duplicates(self) -> None:
        """Two players with same original_round=3, max_per_round=1 → only one kept."""
        candidates = [
            _decision(1, 30.0, original_round=3),
            _decision(2, 25.0, original_round=3),
            _decision(3, 10.0, original_round=5),
        ]
        constraints = KeeperConstraints(
            max_keepers=2,
            max_per_round=1,
        )

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        # Can only keep one from round 3 → player 1 (surplus 30) + player 3 (surplus 10)
        assert ids == {1, 3}

    def test_protected_rounds_excludes_players(self) -> None:
        """Player with original_round=2, escalation=1 → effective round 1 which is protected."""
        candidates = [
            _decision(1, 30.0, original_round=2),  # escalates to round 1 → protected
            _decision(2, 20.0, original_round=5),  # escalates to round 4 → ok
            _decision(3, 15.0, original_round=4),  # escalates to round 3 → ok
        ]
        constraints = KeeperConstraints(
            max_keepers=2,
            round_escalation=1,
            protected_rounds=frozenset({1}),
        )

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        # Player 1 excluded (escalated round 1 is protected), best is player 2 + player 3
        assert 1 not in ids
        assert ids == {2, 3}

    def test_undrafted_round_assigns_round_to_none(self) -> None:
        """Player with original_round=None, undrafted_round=20 → treated as round 20."""
        candidates = [
            _decision(1, 25.0, original_round=None),  # undrafted → round 20
            _decision(2, 20.0, original_round=20),  # also round 20
            _decision(3, 15.0, original_round=5),
        ]
        constraints = KeeperConstraints(
            max_keepers=2,
            max_per_round=1,
            undrafted_round=20,
        )

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        # Player 1 (undrafted→round 20) and player 2 (round 20) conflict on max_per_round=1
        # Best combo: player 1 (surplus 25) + player 3 (surplus 15) = 40
        assert ids == {1, 3}

    def test_no_round_constraints_when_fields_default(self) -> None:
        """All round fields at defaults → existing behavior unchanged."""
        candidates = [
            _decision(1, 30.0, original_round=3),
            _decision(2, 25.0, original_round=3),
            _decision(3, 10.0, original_round=5),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        # No round constraints → top 2 by surplus
        assert ids == {1, 2}

    def test_round_constraints_change_optimal_set(self) -> None:
        """With vs without round constraints produces different optimal sets."""
        candidates = [
            _decision(1, 30.0, original_round=3),
            _decision(2, 25.0, original_round=3),
            _decision(3, 10.0, original_round=5),
        ]

        without_round = KeeperConstraints(max_keepers=2)
        with_round = KeeperConstraints(max_keepers=2, max_per_round=1)

        result_without = solve_keepers(candidates, without_round)
        result_with = solve_keepers(candidates, with_round)

        ids_without = {p.player_id for p in result_without.optimal.players}
        ids_with = {p.player_id for p in result_with.optimal.players}

        assert ids_without != ids_with
        assert ids_without == {1, 2}
        assert ids_with == {1, 3}


# ── Helpers for Phase 2 tests ──────────────────────────────────────────────────


def _valuation(
    player_id: int,
    position: str,
    value: float,
) -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2026,
        system="test",
        version="v1",
        projection_system="test",
        projection_version="v1",
        player_type=PlayerType.BATTER if position != "SP" else PlayerType.PITCHER,
        position=position,
        value=value,
        rank=0,
        category_scores={},
    )


def _league(
    teams: int = 12,
    positions: dict[str, int] | None = None,
    pitcher_positions: dict[str, int] | None = None,
    roster_util: int = 0,
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=teams,
        budget=260,
        roster_batters=14,
        roster_pitchers=9,
        batting_categories=(),
        pitching_categories=(),
        positions=positions or {"C": 1, "SS": 1, "OF": 3},
        pitcher_positions=pitcher_positions or {"SP": 2},
        roster_util=roster_util,
    )


# ── compute_adjusted_draft_pool tests ─────────────────────────────────────────


class TestComputeAdjustedDraftPool:
    def test_removes_kept_players(self) -> None:
        valuations = [
            _valuation(1, "SS", 30.0),
            _valuation(2, "SS", 20.0),
            _valuation(3, "OF", 15.0),
        ]
        league = _league(teams=1)

        pool, _levels = compute_adjusted_draft_pool({1}, valuations, league)

        pool_ids = {v.player_id for v in pool}
        assert 1 not in pool_ids
        assert pool_ids == {2, 3}

    def test_replacement_level_per_position(self) -> None:
        # 2 teams, 1 SS slot each → replacement level = value at rank 2
        valuations = [
            _valuation(1, "SS", 40.0),
            _valuation(2, "SS", 30.0),
            _valuation(3, "SS", 20.0),
            _valuation(4, "SS", 10.0),
        ]
        league = _league(teams=2, positions={"SS": 1})

        _pool, levels = compute_adjusted_draft_pool(set(), valuations, league)

        # rank 2 (0-indexed) = 3rd player = 20.0
        assert levels["SS"] == pytest.approx(20.0)

    def test_replacement_level_changes_when_keepers_removed(self) -> None:
        valuations = [
            _valuation(1, "SS", 40.0),
            _valuation(2, "SS", 30.0),
            _valuation(3, "SS", 20.0),
            _valuation(4, "SS", 10.0),
        ]
        league = _league(teams=2, positions={"SS": 1})

        _pool_before, levels_before = compute_adjusted_draft_pool(set(), valuations, league)
        _pool_after, levels_after = compute_adjusted_draft_pool({1}, valuations, league)

        # Before: replacement = rank 2 of [40, 30, 20, 10] → 20.0
        # After: replacement = rank 2 of [30, 20, 10] → 10.0
        assert levels_before["SS"] == pytest.approx(20.0)
        assert levels_after["SS"] == pytest.approx(10.0)

    def test_position_with_no_pool_entries_gets_zero(self) -> None:
        valuations = [_valuation(1, "OF", 20.0)]
        league = _league(teams=1, positions={"C": 1, "OF": 1})

        _pool, levels = compute_adjusted_draft_pool(set(), valuations, league)

        assert levels["C"] == pytest.approx(0.0)

    def test_returned_pool_sorted_by_value_desc(self) -> None:
        valuations = [
            _valuation(1, "OF", 10.0),
            _valuation(2, "SS", 30.0),
            _valuation(3, "C", 20.0),
        ]
        league = _league(teams=1)

        pool, _levels = compute_adjusted_draft_pool(set(), valuations, league)

        values = [v.value for v in pool]
        assert values == sorted(values, reverse=True)

    def test_case_normalized_positions(self) -> None:
        valuations = [
            _valuation(1, "SS", 40.0),
            _valuation(2, "SS", 30.0),
        ]
        league = _league(teams=1, positions={"SS": 1})

        _pool, levels = compute_adjusted_draft_pool(set(), valuations, league)

        assert "SS" in levels

    def test_includes_pitcher_positions(self) -> None:
        valuations = [
            _valuation(1, "SP", 30.0),
            _valuation(2, "SP", 25.0),
            _valuation(3, "SP", 20.0),
        ]
        league = _league(teams=1, pitcher_positions={"SP": 2})

        _pool, levels = compute_adjusted_draft_pool(set(), valuations, league)

        # 1 team * 2 SP slots → replacement at rank 2 (0-indexed) = 20.0
        assert levels["SP"] == pytest.approx(20.0)


# ── solve_keepers_with_pool tests ─────────────────────────────────────────────


class TestSolveKeepersWithPool:
    def test_pool_effect_changes_optimal_set(self) -> None:
        """When the league keeps many SS, keeping our own SS becomes more valuable."""
        # Without pool: player 1 (OF, surplus 25) beats player 2 (SS, surplus 20)
        # With pool: league keeps 10 SS, depleting SS pool → SS keeper more valuable
        candidates = [
            _decision(1, 25.0, position="OF"),
            _decision(2, 20.0, position="SS"),
        ]
        constraints = KeeperConstraints(max_keepers=1)

        # Without pool awareness
        basic = solve_keepers(candidates, constraints)
        assert basic.optimal.players[0].player_id == 1  # OF wins on surplus alone

        # Build a pool where SS is scarce but OF is deep
        valuations = [_valuation(100 + i, "OF", 30.0 - i * 0.5) for i in range(40)] + [
            _valuation(200 + i, "SS", 28.0 - i * 2.0) for i in range(15)
        ]
        # League keeps 10 of the top SS
        league_keeper_ids = {200 + i for i in range(10)}

        league = _league(
            teams=12,
            positions={"OF": 3, "SS": 1},
            pitcher_positions={},
        )

        result = solve_keepers_with_pool(candidates, constraints, league_keeper_ids, valuations, league)

        # SS should now be more valuable because the pool is depleted
        assert result.optimal.players[0].player_id == 2

    def test_deep_position_reduces_keeper_value(self) -> None:
        """When draft pool is deep at a position, keeping there is less valuable."""
        candidates = [
            _decision(1, 22.0, position="C"),
            _decision(2, 20.0, position="OF"),
        ]
        constraints = KeeperConstraints(max_keepers=1)

        # OF is very deep, C is scarce
        valuations = [_valuation(100 + i, "OF", 25.0 - i * 0.3) for i in range(40)] + [
            _valuation(200 + i, "C", 15.0 - i * 3.0) for i in range(5)
        ]
        league = _league(
            teams=12,
            positions={"C": 1, "OF": 3},
            pitcher_positions={},
        )

        result = solve_keepers_with_pool(candidates, constraints, set(), valuations, league)

        # C is scarce → keeping the catcher is more valuable despite similar surplus
        assert result.optimal.players[0].player_id == 1

    def test_returns_valid_solution_structure(self) -> None:
        candidates = [
            _decision(1, 25.0, position="OF"),
            _decision(2, 20.0, position="SS"),
            _decision(3, 15.0, position="C"),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        valuations = [_valuation(100 + i, "OF", 20.0 - i) for i in range(20)]
        league = _league(teams=2, positions={"OF": 3, "SS": 1, "C": 1}, pitcher_positions={})

        result = solve_keepers_with_pool(candidates, constraints, set(), valuations, league)

        assert result.optimal is not None
        assert len(result.optimal.players) == 2
        assert len(result.sensitivity) > 0
        assert result.optimal.score >= result.optimal.total_surplus

    def test_estimated_draft_value_decreases_with_more_keepers(self) -> None:
        """More keepers → fewer draft slots → lower estimated draft value component."""
        # All candidates at OF so keeping more reduces OF draft slots
        candidates = [
            _decision(1, 30.0, position="OF"),
            _decision(2, 25.0, position="OF"),
            _decision(3, 20.0, position="OF"),
        ]
        valuations = [_valuation(100 + i, "OF", 20.0 - i * 0.5) for i in range(30)]
        league = _league(teams=2, positions={"OF": 5}, pitcher_positions={})

        # Keep 1: 4 unfilled OF slots
        result_1 = solve_keepers_with_pool(
            candidates,
            KeeperConstraints(max_keepers=1),
            set(),
            valuations,
            league,
        )
        # Keep 2: 3 unfilled OF slots
        result_2 = solve_keepers_with_pool(
            candidates,
            KeeperConstraints(max_keepers=2),
            set(),
            valuations,
            league,
        )

        # Draft value component = score - surplus
        draft_val_1 = result_1.optimal.score - result_1.optimal.total_surplus
        draft_val_2 = result_2.optimal.score - result_2.optimal.total_surplus
        assert draft_val_1 > draft_val_2


# ── parse_league_keepers tests ────────────────────────────────────────────────


class TestParseLeagueKeepers:
    def test_matches_player_names(self) -> None:
        players = [
            Player(name_first="Mike", name_last="Trout", id=1),
            Player(name_first="Shohei", name_last="Ohtani", id=2),
        ]
        rows = [{"player_name": "Mike Trout"}, {"player_name": "Shohei Ohtani"}]

        matched, unmatched = parse_league_keepers(rows, players)

        assert matched == {1, 2}
        assert unmatched == []

    def test_returns_unmatched_names(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        rows = [
            {"player_name": "Mike Trout"},
            {"player_name": "Nobody Real"},
        ]

        matched, unmatched = parse_league_keepers(rows, players)

        assert matched == {1}
        assert unmatched == ["Nobody Real"]

    def test_case_insensitive(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]
        rows = [{"player_name": "mike trout"}]

        matched, unmatched = parse_league_keepers(rows, players)

        assert matched == {1}

    def test_empty_rows(self) -> None:
        players = [Player(name_first="Mike", name_last="Trout", id=1)]

        matched, unmatched = parse_league_keepers([], players)

        assert matched == set()
        assert unmatched == []


# ── compare_scenarios tests ──────────────────────────────────────────────────


class TestCompareScenarios:
    def test_ranks_scenarios_by_score_descending(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 10.0),
            _decision(4, 5.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        scenarios = [
            ("worst", [3, 4]),
            ("best", [1, 2]),
            ("middle", [1, 3]),
        ]

        result = compare_scenarios(scenarios, candidates, constraints)

        assert [s.name for s in result] == ["best", "middle", "worst"]

    def test_delta_vs_optimal_zero_for_best(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        scenarios = [
            ("best", [1, 2]),
            ("worse", [1, 3]),
        ]

        result = compare_scenarios(scenarios, candidates, constraints)

        assert result[0].delta_vs_optimal == pytest.approx(0.0)
        assert result[1].delta_vs_optimal == pytest.approx(10.0)

    def test_single_scenario(self) -> None:
        candidates = [
            _decision(1, 20.0),
            _decision(2, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        scenarios = [("only", [1, 2])]

        result = compare_scenarios(scenarios, candidates, constraints)

        assert len(result) == 1
        assert result[0].delta_vs_optimal == pytest.approx(0.0)

    def test_invalid_player_id_raises(self) -> None:
        candidates = [_decision(1, 20.0)]
        constraints = KeeperConstraints(max_keepers=1)
        scenarios = [("bad", [99])]

        with pytest.raises(ValueError, match="player_id=99"):
            compare_scenarios(scenarios, candidates, constraints)

    def test_keepers_list_matches_input(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        scenarios = [("test", [2, 1])]

        result = compare_scenarios(scenarios, candidates, constraints)

        assert result[0].keepers == [2, 1]


# ── keeper_trade_impact tests ────────────────────────────────────────────────


class TestKeeperTradeImpact:
    def test_acquiring_high_surplus_player_improves_score(self) -> None:
        candidates = [
            _decision(1, 20.0),
            _decision(2, 15.0),
            _decision(3, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        new_player = _decision(4, 30.0)

        result = keeper_trade_impact(candidates, constraints, acquire=[new_player], release=[3])

        assert result.score_delta > 0
        after_ids = {p.player_id for p in result.after.optimal.players}
        assert 4 in after_ids

    def test_releasing_valuable_player_hurts_score(self) -> None:
        candidates = [
            _decision(1, 30.0),
            _decision(2, 20.0),
            _decision(3, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = keeper_trade_impact(candidates, constraints, acquire=[], release=[1])

        assert result.score_delta < 0

    def test_score_delta_is_correct(self) -> None:
        candidates = [
            _decision(1, 20.0),
            _decision(2, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        new_player = _decision(3, 25.0)

        result = keeper_trade_impact(candidates, constraints, acquire=[new_player], release=[2])

        expected_delta = result.after.optimal.score - result.before.optimal.score
        assert result.score_delta == pytest.approx(expected_delta)

    def test_release_nonexistent_player_raises(self) -> None:
        candidates = [_decision(1, 20.0)]
        constraints = KeeperConstraints(max_keepers=1)

        with pytest.raises(ValueError, match="player_id=99"):
            keeper_trade_impact(candidates, constraints, acquire=[], release=[99])

    def test_before_and_after_are_valid_solutions(self) -> None:
        candidates = [
            _decision(1, 25.0),
            _decision(2, 15.0),
            _decision(3, 10.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)
        new_player = _decision(4, 20.0)

        result = keeper_trade_impact(candidates, constraints, acquire=[new_player], release=[3])

        assert result.before.optimal is not None
        assert result.after.optimal is not None
        assert len(result.before.optimal.players) == 2
        assert len(result.after.optimal.players) == 2
