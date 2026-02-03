from fantasy_baseball_manager.keeper.replacement import DraftPoolReplacementCalculator
from fantasy_baseball_manager.valuation.models import PlayerValue


def _make_pool(count: int) -> list[PlayerValue]:
    """Create a pool of players with descending values."""
    return [
        PlayerValue(
            player_id=f"p{i}",
            name=f"Player {i}",
            category_values=(),
            total_value=100.0 - i,
        )
        for i in range(count)
    ]


class TestDraftPoolReplacementCalculator:
    def test_slot_costs_decrease_monotonically(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        pool = _make_pool(50)
        costs = calc.compute_slot_costs(pool, num_teams=12, num_keeper_slots=4)
        assert len(costs) == 4
        for i in range(len(costs) - 1):
            assert costs[i].replacement_value > costs[i + 1].replacement_value

    def test_slot_numbers_are_sequential(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        pool = _make_pool(50)
        costs = calc.compute_slot_costs(pool, num_teams=12, num_keeper_slots=4)
        assert [sc.slot_number for sc in costs] == [1, 2, 3, 4]

    def test_pick_position_affects_costs(self) -> None:
        pool = _make_pool(50)
        calc_first = DraftPoolReplacementCalculator(user_pick_position=1)
        calc_last = DraftPoolReplacementCalculator(user_pick_position=12)
        costs_first = calc_first.compute_slot_costs(pool, num_teams=12, num_keeper_slots=4)
        costs_last = calc_last.compute_slot_costs(pool, num_teams=12, num_keeper_slots=4)
        # Pick 1 gets the best player in round 1; pick 12 gets the worst
        assert costs_first[0].replacement_value > costs_last[0].replacement_value
        # But in round 2 (snake), pick 12 goes first
        assert costs_last[1].replacement_value > costs_first[1].replacement_value

    def test_snake_order_respected(self) -> None:
        """With 3 teams, pick 1: round 1 pick 0, round 2 pick 2, round 3 pick 0, round 4 pick 2."""
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        pool = _make_pool(20)
        costs = calc.compute_slot_costs(pool, num_teams=3, num_keeper_slots=4)
        # Pick indices for team 0 in snake(3 teams, 4 rounds): [0, 5, 6, 11]
        # Snake: round 1: [0,1,2], round 2: [2,1,0], round 3: [0,1,2], round 4: [2,1,0]
        # Team 0 picks at positions: 0, 5, 6, 11
        assert costs[0].replacement_value == pool[0].total_value
        assert costs[1].replacement_value == pool[5].total_value
        assert costs[2].replacement_value == pool[6].total_value
        assert costs[3].replacement_value == pool[11].total_value

    def test_removing_keepers_shifts_replacement_values(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        full_pool = _make_pool(50)
        # Remove the top 5 players (simulating them being kept)
        depleted_pool = full_pool[5:]
        costs_full = calc.compute_slot_costs(full_pool, num_teams=12, num_keeper_slots=4)
        costs_depleted = calc.compute_slot_costs(depleted_pool, num_teams=12, num_keeper_slots=4)
        # With top players removed, replacement values should be lower
        for full, depleted in zip(costs_full, costs_depleted, strict=True):
            assert full.replacement_value > depleted.replacement_value

    def test_empty_pool(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        costs = calc.compute_slot_costs([], num_teams=12, num_keeper_slots=4)
        assert len(costs) == 4
        assert all(sc.replacement_value == 0.0 for sc in costs)

    def test_pool_smaller_than_total_picks(self) -> None:
        calc = DraftPoolReplacementCalculator(user_pick_position=1)
        # 3 players for 12 teams * 4 rounds = 48 picks
        pool = _make_pool(3)
        costs = calc.compute_slot_costs(pool, num_teams=12, num_keeper_slots=4)
        assert len(costs) == 4
        # First slot gets a value, rest are 0
        assert costs[0].replacement_value == pool[0].total_value
        assert all(sc.replacement_value == 0.0 for sc in costs[1:])
