"""Tests for availability window computation."""

from __future__ import annotations

from fantasy_baseball_manager.services.draft_plan import (
    _user_pick_numbers,
    compute_availability_windows,
    compute_player_availability_curve,
)

NAMES: dict[int, str] = {1: "Alice", 2: "Bob", 3: "Charlie"}
POSITIONS: dict[int, str] = {1: "SP", 2: "OF", 3: "1B"}


class TestTightWindow:
    """Player always drafted at pick 5 → tight earliest/median/latest."""

    def test_tight_window(self) -> None:
        picks = {1: [5] * 100}
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=100, user_next_pick=6)
        assert len(windows) == 1
        w = windows[0]
        assert w.player_id == 1
        assert w.earliest_pick == 5.0
        assert w.median_pick == 5.0
        assert w.latest_pick == 5.0


class TestWideWindow:
    """Picks spread 10-50 → wide earliest-latest range."""

    def test_wide_window(self) -> None:
        # 100 picks evenly spread from 10 to 50
        picks = {1: list(range(10, 51)) * 2 + list(range(10, 19))}  # 82+18=100? let's just do 100
        spread = list(range(10, 51))  # 41 values
        # Repeat to get 100 data points
        pick_list = (spread * 3)[:100]
        picks = {1: pick_list}
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=100, user_next_pick=30)
        w = windows[0]
        assert w.earliest_pick < w.median_pick < w.latest_pick
        # With spread 10-50, earliest (5th pct) should be near low end, latest near high end
        assert w.earliest_pick < 20
        assert w.latest_pick > 40


class TestAvailableAtUserPickAllBefore:
    """All picks < user pick → availability = 0.0."""

    def test_all_drafted_before_user_pick(self) -> None:
        """All 5 sims draft this player before pick 10 → 0% available."""
        picks = {1: [1, 2, 3, 4, 5]}
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=5, user_next_pick=10)
        # All 5 picks are < 10, so all drafted before user pick
        assert windows[0].available_at_user_pick == 0.0


class TestAvailableAtUserPickAllAfter:
    """All picks > user pick → availability = 1.0."""

    def test_all_after(self) -> None:
        picks = {1: [20, 25, 30, 35, 40]}
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=5, user_next_pick=10)
        assert windows[0].available_at_user_pick == 1.0


class TestAvailableAtUserPickMixed:
    """Partial availability: some before, some after."""

    def test_mixed_availability(self) -> None:
        # 3 sims draft at pick 5 (before 10), 7 sims draft at pick 15 (after 10)
        picks = {1: [5, 5, 5, 15, 15, 15, 15, 15, 15, 15]}
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=10, user_next_pick=10)
        # 3 drafted before pick 10, so 7/10 = 0.7 available
        assert windows[0].available_at_user_pick == 0.7


class TestUndraftedSimsCountAsAvailable:
    """100 sims, 60 picks → 40 undrafted count as available."""

    def test_undrafted_available(self) -> None:
        # 60 sims draft this player at pick 5 (before pick 10)
        # 40 sims don't draft them at all
        picks = {1: [5] * 60}
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=100, user_next_pick=10)
        # 60 drafted before pick 10, 40 undrafted → (100 - 60) / 100 = 0.4
        assert windows[0].available_at_user_pick == 0.4

    def test_undrafted_all_available(self) -> None:
        # 60 sims draft at pick 15 (after pick 10), 40 undrafted
        picks = {1: [15] * 60}
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=100, user_next_pick=10)
        # 0 drafted before pick 10, so (100 - 0) / 100 = 1.0
        assert windows[0].available_at_user_pick == 1.0


class TestUserPickNumbers:
    """Verify snake draft math."""

    def test_slot_0_4_teams_3_rounds(self) -> None:
        picks = _user_pick_numbers(slot=0, teams=4, total_rounds=3)
        # Round 1 (odd): (0)*4 + 0 + 1 = 1
        # Round 2 (even): 2*4 - 0 = 8
        # Round 3 (odd): (2)*4 + 0 + 1 = 9
        assert picks == [1, 8, 9]

    def test_slot_3_4_teams_3_rounds(self) -> None:
        picks = _user_pick_numbers(slot=3, teams=4, total_rounds=3)
        # Round 1 (odd): (0)*4 + 3 + 1 = 4
        # Round 2 (even): 2*4 - 3 = 5
        # Round 3 (odd): (2)*4 + 3 + 1 = 12
        assert picks == [4, 5, 12]

    def test_slot_0_12_teams_2_rounds(self) -> None:
        picks = _user_pick_numbers(slot=0, teams=12, total_rounds=2)
        # Round 1: 1
        # Round 2: 24
        assert picks == [1, 24]


class TestCurveMonotonicallyNonIncreasing:
    """Availability can only decrease as picks advance."""

    def test_monotonic(self) -> None:
        # Player drafted at various picks
        picks = {1: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
        curve = compute_player_availability_curve(
            player_id=1,
            all_player_picks=picks,
            player_names=NAMES,
            player_positions=POSITIONS,
            n_simulations=10,
            slot=0,
            teams=12,
            total_rounds=5,
        )
        probs = [pa.probability for pa in curve.pick_availabilities]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1], f"Not non-increasing: {probs[i]} < {probs[i + 1]} at index {i}"


class TestCurveLength:
    """Curve length should equal total_rounds."""

    def test_length_equals_total_rounds(self) -> None:
        picks = {1: [5]}
        curve = compute_player_availability_curve(
            player_id=1,
            all_player_picks=picks,
            player_names=NAMES,
            player_positions=POSITIONS,
            n_simulations=1,
            slot=0,
            teams=12,
            total_rounds=7,
        )
        assert len(curve.pick_availabilities) == 7


class TestSortedByMedian:
    """Windows should be sorted by median_pick."""

    def test_sorted_output(self) -> None:
        picks = {
            1: [30] * 10,  # median 30
            2: [10] * 10,  # median 10
            3: [20] * 10,  # median 20
        }
        windows = compute_availability_windows(picks, NAMES, POSITIONS, n_simulations=10, user_next_pick=5)
        medians = [w.median_pick for w in windows]
        assert medians == sorted(medians)
