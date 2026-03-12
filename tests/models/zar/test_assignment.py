import time

import pytest

from fantasy_baseball_manager.models.zar.assignment import assign_positions


class TestAssignPositionsEmpty:
    def test_empty_input(self) -> None:
        result = assign_positions([], [], {}, num_teams=12)
        assert result.assignments == {}
        assert result.replacement == {}
        assert result.var_values == []


class TestAssignPositionsSinglePosition:
    def test_single_position_simple(self) -> None:
        """5 players, 1 position with 1 slot, 4 teams → 4 assigned."""
        scores = [10.0, 8.0, 6.0, 4.0, 2.0]
        positions = [["C"]] * 5
        roster_spots = {"C": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=4)

        # Top 4 should be assigned
        assert len(result.assignments) == 4
        for i in range(4):
            assert i in result.assignments
        assert 4 not in result.assignments

        # Replacement = worst assigned player's score = 4.0
        assert result.replacement["C"] == pytest.approx(4.0)

        # VAR for assigned players
        assert result.var_values[0] == pytest.approx(6.0)  # 10 - 4
        assert result.var_values[1] == pytest.approx(4.0)  # 8 - 4
        assert result.var_values[2] == pytest.approx(2.0)  # 6 - 4
        assert result.var_values[3] == pytest.approx(0.0)  # 4 - 4 (replacement player)
        assert result.var_values[4] == pytest.approx(0.0)  # unassigned


class TestAssignPositionsMultiPositionNoFlex:
    def test_disjoint_positions(self) -> None:
        """Disjoint positions with no flex — matches greedy behavior."""
        scores = [10.0, 8.0, 6.0, 4.0]
        positions = [["SP"], ["SP"], ["RP"], ["RP"]]
        roster_spots = {"SP": 1, "RP": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=1)

        assert result.assignments[0] == "SP"
        assert result.assignments[2] == "RP"
        assert len(result.assignments) == 2
        assert result.replacement["SP"] == pytest.approx(10.0)
        assert result.replacement["RP"] == pytest.approx(6.0)


class TestAssignPositionsFlexSlot:
    def test_flex_slot_optimal_assignment(self) -> None:
        """SP + P slots: inflated SP replacement includes P-overflow SPs."""
        # 6 pitchers: all eligible for SP and P
        scores = [10.0, 8.0, 6.0, 5.0, 3.0, 1.0]
        positions = [["SP", "P"]] * 6
        roster_spots = {"SP": 1, "P": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=2)

        # 4 total slots (2 SP + 2 P)
        assert len(result.assignments) == 4

        # The solver should assign top 4 players optimally
        for i in range(4):
            assert i in result.assignments

        sp_assigned = [i for i, pos in result.assignments.items() if pos == "SP"]
        p_assigned = [i for i, pos in result.assignments.items() if pos == "P"]
        assert len(sp_assigned) == 2
        assert len(p_assigned) == 2

        # SP replacement is inflated: includes P-overflow SPs (all 4 are SP-primary)
        # so SP replacement = worst of all 4 assigned = score of player 3 = 5.0
        assert result.replacement["SP"] == pytest.approx(5.0)

        # P replacement = worst P-assigned player (unchanged)
        p_min = min(scores[i] for i in p_assigned)
        assert result.replacement["P"] == pytest.approx(p_min)

        # No cliff: SP-assigned and P-assigned players both use SP replacement
        # Player 0 (SP, score 10): VAR = 10 - 5 = 5
        # Player 2 (P, score 6): VAR = 6 - 5 = 1 (not 6 - 5 = 1 vs old 6 - 5 = 1)
        # The key: adjacent players straddle the SP/P boundary smoothly
        for idx in result.assignments:
            assert result.var_values[idx] >= -1e-9


class TestAssignPositionsUtil:
    def test_util_slot_assignment(self) -> None:
        """Batter positions + UTIL: scarce-position players fill specific slots first."""
        # 2 catchers, 3 outfielders; 1 C slot + 1 UTIL slot, 1 team
        # C is scarcer, so catchers should fill C, OF fills UTIL
        scores = [10.0, 7.0, 8.0, 6.0, 4.0]
        positions = [
            ["C", "UTIL"],
            ["C", "UTIL"],
            ["OF", "UTIL"],
            ["OF", "UTIL"],
            ["OF", "UTIL"],
        ]
        roster_spots = {"C": 1, "OF": 1, "UTIL": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=1)

        # 3 total slots
        assert len(result.assignments) == 3

        # The C slot should have a catcher
        c_assigned = [i for i, pos in result.assignments.items() if pos == "C"]
        assert len(c_assigned) == 1
        assert c_assigned[0] in (0, 1)  # one of the catchers


class TestAssignPositionsUtilOnlyElite:
    def test_util_only_elite_player_assigned(self) -> None:
        """An elite UTIL-only player must be assigned even with many candidates.

        Regression test for a numerical issue where 1e18 penalty caused
        linear_sum_assignment to misassign the top player to an ineligible slot.
        """
        # 200 players, 100 slots: realistic scale
        n = 200
        scores = [float(n - i) for i in range(n)]
        positions: list[list[str]] = []
        for i in range(n):
            if i == 0:
                # Elite UTIL-only player (like Ohtani as DH)
                positions.append(["UTIL"])
            elif i % 10 == 0:
                positions.append(["C", "UTIL"])
            elif i % 3 == 0:
                positions.append(["OF", "UTIL"])
            elif i % 3 == 1:
                positions.append(["SS", "UTIL"])
            else:
                positions.append(["1B", "UTIL"])
        roster_spots = {"C": 1, "1B": 1, "SS": 1, "OF": 3, "UTIL": 1}
        num_teams = 10  # 70 total slots

        result = assign_positions(scores, positions, roster_spots, num_teams=num_teams)

        # The elite UTIL-only player MUST be assigned
        assert 0 in result.assignments, "Elite UTIL-only player should be assigned"
        assert result.assignments[0] == "UTIL"
        assert result.var_values[0] > 0


class TestAssignPositionsOptimalBeatsGreedy:
    def test_optimal_beats_greedy(self) -> None:
        """Construct a scenario where greedy assignment is suboptimal.

        Player 0: score=10, eligible=[SP, P]
        Player 1: score=9,  eligible=[P]  (P-only)
        Player 2: score=8,  eligible=[SP, P]
        Slots: 1 SP, 1 P (1 team)

        Greedy (each picks lowest replacement): might assign both dual-eligible
        to P, leaving SP empty. Optimal: Player 0→SP, Player 1→P.
        """
        scores = [10.0, 9.0, 8.0]
        positions = [["SP", "P"], ["P"], ["SP", "P"]]
        roster_spots = {"SP": 1, "P": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=1)

        # Optimal: assign 2 players (2 slots)
        assert len(result.assignments) == 2

        # Player 1 must be assigned to P (only eligible slot)
        assert result.assignments[1] == "P"

        # Player 0 should be assigned to SP (maximizes total score)
        assert result.assignments[0] == "SP"


class TestAssignPositionsEdgeCases:
    def test_fewer_players_than_slots(self) -> None:
        """All players assigned when fewer than total slots."""
        scores = [10.0, 8.0]
        positions = [["C"], ["C"]]
        roster_spots = {"C": 3}
        result = assign_positions(scores, positions, roster_spots, num_teams=2)

        # Only 2 players but 6 slots — both should be assigned
        assert len(result.assignments) == 2
        assert 0 in result.assignments
        assert 1 in result.assignments

    def test_no_eligible_players_for_position(self) -> None:
        """Position with no eligible players gets replacement = 0.0."""
        scores = [10.0]
        positions = [["SP"]]
        roster_spots = {"SP": 1, "RP": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=1)

        assert result.replacement.get("RP", 0.0) == pytest.approx(0.0)

    def test_var_values_length_matches_input(self) -> None:
        """Output VAR list always has same length as input."""
        for n in (0, 1, 5, 20):
            scores = [float(i) for i in range(n)]
            positions = [["C"]] * n
            roster_spots = {"C": 1}
            result = assign_positions(scores, positions, roster_spots, num_teams=1)
            assert len(result.var_values) == n


class TestAssignPositionsInvariants:
    def test_assigned_count_equals_slots(self) -> None:
        """Exactly total_slots players assigned when enough players exist."""
        scores = [float(20 - i) for i in range(20)]
        positions = [["SP", "P"]] * 10 + [["RP", "P"]] * 10
        roster_spots = {"SP": 1, "RP": 1, "P": 2}
        num_teams = 2
        result = assign_positions(scores, positions, roster_spots, num_teams=num_teams)

        total_slots = sum(v * num_teams for v in roster_spots.values())  # 8
        assert len(result.assignments) == total_slots

    def test_all_assigned_have_nonneg_var(self) -> None:
        """Every assigned player has VAR >= 0."""
        scores = [10.0, 8.0, 6.0, 4.0, 2.0]
        positions = [["SP", "P"]] * 5
        roster_spots = {"SP": 1, "P": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=1)

        for idx in result.assignments:
            assert result.var_values[idx] >= -1e-9

    def test_replacement_player_has_zero_var(self) -> None:
        """Worst player at each position has VAR = 0."""
        scores = [10.0, 8.0, 6.0, 4.0]
        positions = [["C"]] * 4
        roster_spots = {"C": 1}
        result = assign_positions(scores, positions, roster_spots, num_teams=3)

        # 3 assigned, replacement = player with score 6.0
        c_assigned = [i for i, pos in result.assignments.items() if pos == "C"]
        worst_score = min(scores[i] for i in c_assigned)
        worst_idx = next(i for i in c_assigned if scores[i] == worst_score)
        assert result.var_values[worst_idx] == pytest.approx(0.0)


class TestAssignPositionsPerformance:
    @pytest.mark.slow
    def test_performance_1000_candidates_100_slots(self) -> None:
        """Solver completes in under 1 second for 1000 candidates × 100 slots."""
        n_players = 1000
        scores = [float(n_players - i) for i in range(n_players)]
        # Mix of positions
        positions = []
        for i in range(n_players):
            if i % 3 == 0:
                positions.append(["SP", "P"])
            elif i % 3 == 1:
                positions.append(["RP", "P"])
            else:
                positions.append(["SP", "RP", "P"])
        roster_spots = {"SP": 3, "RP": 3, "P": 4}
        num_teams = 10  # 100 total slots

        start = time.monotonic()
        result = assign_positions(scores, positions, roster_spots, num_teams=num_teams)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        assert len(result.assignments) == 100
        assert len(result.var_values) == n_players
