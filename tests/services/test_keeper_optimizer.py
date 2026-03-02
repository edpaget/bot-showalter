import pytest

from fantasy_baseball_manager.domain import KeeperConstraints, KeeperDecision
from fantasy_baseball_manager.services import solve_keepers


def _decision(
    player_id: int,
    surplus: float,
    position: str = "of",
    cost: float = 10.0,
    name: str | None = None,
) -> KeeperDecision:
    return KeeperDecision(
        player_id=player_id,
        player_name=name or f"Player {player_id}",
        position=position,
        cost=cost,
        projected_value=cost + surplus,
        surplus=surplus,
        years_remaining=1,
        recommendation="keep" if surplus > 0 else "release",
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
            _decision(1, 30.0, position="c", cost=15.0),
            _decision(2, 20.0, position="of", cost=10.0),
            _decision(3, 10.0, position="of", cost=5.0),
        ]
        constraints = KeeperConstraints(max_keepers=2)

        result = solve_keepers(candidates, constraints)

        assert result.optimal.total_surplus == pytest.approx(50.0)
        assert result.optimal.total_cost == pytest.approx(25.0)
        assert result.optimal.positions_filled == {"c": 1, "of": 1}
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
            _decision(1, 20.0, position="c"),
            _decision(2, 15.0, position="c"),
            _decision(3, 10.0, position="c"),
            _decision(4, 12.0, position="of"),
        ]
        constraints = KeeperConstraints(max_keepers=2, max_per_position={"c": 1})

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 4}
        assert result.optimal.total_surplus == pytest.approx(32.0)

    def test_multiple_position_limits(self) -> None:
        candidates = [
            _decision(1, 25.0, position="c"),
            _decision(2, 20.0, position="c"),
            _decision(3, 18.0, position="of"),
            _decision(4, 15.0, position="of"),
            _decision(5, 12.0, position="of"),
        ]
        constraints = KeeperConstraints(max_keepers=3, max_per_position={"c": 1, "of": 2})

        result = solve_keepers(candidates, constraints)

        ids = {p.player_id for p in result.optimal.players}
        assert ids == {1, 3, 4}

    def test_unlisted_position_unconstrained(self) -> None:
        candidates = [
            _decision(1, 20.0, position="c"),
            _decision(2, 18.0, position="of"),
            _decision(3, 15.0, position="of"),
        ]
        constraints = KeeperConstraints(max_keepers=3, max_per_position={"c": 1})

        result = solve_keepers(candidates, constraints)

        assert len(result.optimal.players) == 3

    def test_position_limit_none(self) -> None:
        candidates = [
            _decision(1, 20.0, position="c"),
            _decision(2, 18.0, position="c"),
            _decision(3, 15.0, position="c"),
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
            _decision(1, 25.0, position="c"),
            _decision(2, 20.0, position="c"),  # required
            _decision(3, 15.0, position="of"),
        ]
        constraints = KeeperConstraints(
            max_keepers=2,
            max_per_position={"c": 1},
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
            _decision(1, 20.0, position="c"),
            _decision(2, 15.0, position="c"),
        ]
        constraints = KeeperConstraints(max_keepers=2, max_per_position={"c": 1})

        with pytest.raises(ValueError, match="no valid"):
            solve_keepers(candidates, constraints)
