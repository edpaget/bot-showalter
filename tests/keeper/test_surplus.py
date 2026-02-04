import pytest

from fantasy_baseball_manager.keeper.models import KeeperCandidate
from fantasy_baseball_manager.keeper.replacement import DraftPoolReplacementCalculator
from fantasy_baseball_manager.keeper.surplus import SurplusCalculator
from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue, StatCategory


def _pv(player_id: str, name: str, total_value: float) -> PlayerValue:
    return PlayerValue(
        player_id=player_id,
        name=name,
        category_values=(
            CategoryValue(category=StatCategory.HR, raw_stat=20.0, value=total_value / 2),
            CategoryValue(category=StatCategory.R, raw_stat=80.0, value=total_value / 2),
        ),
        total_value=total_value,
    )


def _candidate(player_id: str, name: str, total_value: float, positions: tuple[str, ...] = ("OF",)) -> KeeperCandidate:
    return KeeperCandidate(
        player_id=player_id,
        name=name,
        player_value=_pv(player_id, name, total_value),
        eligible_positions=positions,
    )


def _build_pool(count: int, start_value: float = 50.0) -> list[PlayerValue]:
    """Build a pool of players with descending values starting from start_value."""
    return [_pv(f"pool{i}", f"Pool Player {i}", start_value - i) for i in range(count)]


class TestRankCandidates:
    def test_candidates_ranked_by_surplus(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        candidates = [
            _candidate("c1", "Star", 80.0),
            _candidate("c2", "Good", 60.0),
            _candidate("c3", "Average", 40.0),
        ]
        pool = _build_pool(60) + [c.player_value for c in candidates]

        ranked = surplus.rank_candidates(candidates, pool, other_keepers=set())
        # Sorted by surplus (player_value - replacement at assigned slot), not raw value
        surpluses = [r.surplus_value for r in ranked]
        assert surpluses == sorted(surpluses, reverse=True)
        # All three candidates should be present
        assert {r.player_id for r in ranked} == {"c1", "c2", "c3"}

    def test_negative_surplus_for_bad_candidates(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        # Candidate worse than what you'd draft
        candidates = [_candidate("c1", "Bad Player", -5.0)]
        pool = [*_build_pool(60), candidates[0].player_value]

        ranked = surplus.rank_candidates(candidates, pool, other_keepers=set())
        assert ranked[0].surplus_value < 0

    def test_other_keepers_affect_replacement_values(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        candidates = [_candidate("c1", "Candidate", 55.0)]
        pool = [*_build_pool(60), candidates[0].player_value]

        # Without other keepers
        ranked_no_keepers = surplus.rank_candidates(candidates, pool, other_keepers=set())

        # With other keepers removing top pool players
        other_keepers = {f"pool{i}" for i in range(10)}
        ranked_with_keepers = surplus.rank_candidates(candidates, pool, other_keepers=other_keepers)

        # When top players are kept by others, replacement values drop, so surplus increases
        assert ranked_with_keepers[0].surplus_value > ranked_no_keepers[0].surplus_value


class TestFindOptimalKeepers:
    def test_obvious_keepers_selected(self) -> None:
        """Four elite players among mediocre ones should all be recommended."""
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        elite = [
            _candidate("e1", "Elite 1", 90.0),
            _candidate("e2", "Elite 2", 85.0),
            _candidate("e3", "Elite 3", 80.0),
            _candidate("e4", "Elite 4", 75.0),
        ]
        mediocre = [
            _candidate("m1", "Mediocre 1", 10.0),
            _candidate("m2", "Mediocre 2", 8.0),
        ]
        candidates = elite + mediocre
        pool = _build_pool(60) + [c.player_value for c in candidates]

        result = surplus.find_optimal_keepers(candidates, pool, other_keepers=set())
        keeper_ids = {k.player_id for k in result.keepers}
        assert keeper_ids == {"e1", "e2", "e3", "e4"}

    def test_total_surplus_matches_sum(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        candidates = [
            _candidate("c1", "A", 70.0),
            _candidate("c2", "B", 60.0),
            _candidate("c3", "C", 50.0),
            _candidate("c4", "D", 40.0),
            _candidate("c5", "E", 30.0),
        ]
        pool = _build_pool(60) + [c.player_value for c in candidates]

        result = surplus.find_optimal_keepers(candidates, pool, other_keepers=set())
        assert result.total_surplus == pytest.approx(sum(k.surplus_value for k in result.keepers))

    def test_optimal_with_other_keepers(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        candidates = [
            _candidate("c1", "A", 70.0),
            _candidate("c2", "B", 60.0),
            _candidate("c3", "C", 50.0),
            _candidate("c4", "D", 40.0),
        ]
        pool = _build_pool(60) + [c.player_value for c in candidates]

        # Without other keepers
        result_no = surplus.find_optimal_keepers(candidates, pool, other_keepers=set())
        # With other keepers
        other_keepers = {f"pool{i}" for i in range(5)}
        result_with = surplus.find_optimal_keepers(candidates, pool, other_keepers=other_keepers)

        # Same keepers should be selected (still the best 4), but surplus changes
        assert {k.player_id for k in result_no.keepers} == {k.player_id for k in result_with.keepers}
        assert result_with.total_surplus != result_no.total_surplus

    def test_all_candidates_includes_non_optimal(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        candidates = [
            _candidate("c1", "A", 70.0),
            _candidate("c2", "B", 60.0),
            _candidate("c3", "C", 50.0),
            _candidate("c4", "D", 40.0),
            _candidate("c5", "E", 5.0),
        ]
        pool = _build_pool(60) + [c.player_value for c in candidates]

        result = surplus.find_optimal_keepers(candidates, pool, other_keepers=set())
        all_ids = {c.player_id for c in result.all_candidates}
        assert all_ids == {"c1", "c2", "c3", "c4", "c5"}

    def test_fewer_candidates_than_slots(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        surplus = SurplusCalculator(calc, num_teams=12, num_keeper_slots=4)

        candidates = [
            _candidate("c1", "A", 70.0),
            _candidate("c2", "B", 60.0),
        ]
        pool = _build_pool(60) + [c.player_value for c in candidates]

        result = surplus.find_optimal_keepers(candidates, pool, other_keepers=set())
        assert len(result.keepers) == 2
