from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.mock_draft import BatchSimulationResult
from fantasy_baseball_manager.domain.position import Position
from fantasy_baseball_manager.domain.tier import PlayerTier
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.roster_optimizer import (
    _reduce_league_for_keepers,
    _snake_pick_numbers,
    optimize_auction_budget,
    plan_snake_draft,
    simulate_drafts,
)


def _league(
    budget: int = 260,
    positions: dict[str, int] | None = None,
    pitcher_positions: dict[str, int] | None = None,
    roster_util: int = 0,
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=budget,
        roster_batters=9,
        roster_pitchers=8,
        batting_categories=(
            CategoryConfig(key="hr", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="w", name="W", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        roster_util=roster_util,
        positions=positions or {Position.C: 1, Position.FIRST_BASE: 1, Position.SS: 1, Position.OF: 3},
        pitcher_positions=pitcher_positions or {Position.SP: 2, Position.RP: 2},
    )


def _valuation(player_id: int, position: str, value: float, player_type: str = "batter") -> Valuation:
    return Valuation(
        player_id=player_id,
        season=2025,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="1.0",
        player_type=player_type,
        position=position,
        value=value,
        rank=1,
        category_scores={},
    )


def _make_valuations() -> tuple[list[Valuation], dict[int, str]]:
    """Build a set of valuations with known values for testing."""
    valuations: list[Valuation] = []
    names: dict[int, str] = {}
    pid = 1

    # Catchers: 3 players, descending values
    for val, name in [(30.0, "Catcher A"), (15.0, "Catcher B"), (5.0, "Catcher C")]:
        valuations.append(_valuation(pid, "c", val))
        names[pid] = name
        pid += 1

    # 1B: 3 players
    for val, name in [(25.0, "First A"), (20.0, "First B"), (8.0, "First C")]:
        valuations.append(_valuation(pid, "1b", val))
        names[pid] = name
        pid += 1

    # SS: 3 players
    for val, name in [(35.0, "Short A"), (18.0, "Short B"), (6.0, "Short C")]:
        valuations.append(_valuation(pid, "ss", val))
        names[pid] = name
        pid += 1

    # OF: 6 players (need 3 slots)
    for val, name in [
        (40.0, "OF A"),
        (32.0, "OF B"),
        (28.0, "OF C"),
        (15.0, "OF D"),
        (10.0, "OF E"),
        (3.0, "OF F"),
    ]:
        valuations.append(_valuation(pid, "of", val))
        names[pid] = name
        pid += 1

    # SP: 5 players
    for val, name in [
        (38.0, "SP A"),
        (30.0, "SP B"),
        (20.0, "SP C"),
        (12.0, "SP D"),
        (5.0, "SP E"),
    ]:
        valuations.append(_valuation(pid, "sp", val, player_type="pitcher"))
        names[pid] = name
        pid += 1

    # RP: 4 players
    for val, name in [(22.0, "RP A"), (14.0, "RP B"), (7.0, "RP C"), (2.0, "RP D")]:
        valuations.append(_valuation(pid, "rp", val, player_type="pitcher"))
        names[pid] = name
        pid += 1

    return valuations, names


class TestBudgetSumsToLeagueBudget:
    def test_balanced(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        total = sum(a.budget for a in result)
        assert total == league.budget

    def test_stars_and_scrubs(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="stars_and_scrubs")
        total = sum(a.budget for a in result)
        assert total == league.budget


class TestAllRosterSlotsAccountedFor:
    def test_slot_count(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names)
        # c=1, 1b=1, ss=1, of=3, sp=2, rp=2 = 10
        assert len(result) == 10

    def test_with_util(self) -> None:
        league = _league(roster_util=1)
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names)
        assert len(result) == 11


class TestMinimumBidFloor:
    def test_every_allocation_at_least_one(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        for strategy in ("balanced", "stars_and_scrubs"):
            result = optimize_auction_budget(valuations, league, names, strategy=strategy)
            for alloc in result:
                assert alloc.budget >= 1.0, f"{alloc.position} budget {alloc.budget} < 1 ({strategy})"


class TestStarsAndScrubsConcentrates:
    def test_top_5_over_half(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="stars_and_scrubs")
        sorted_allocs = sorted(result, key=lambda a: a.budget, reverse=True)
        top_5 = sum(a.budget for a in sorted_allocs[:5])
        assert top_5 > league.budget * 0.5


class TestBalancedWithin2x:
    def test_non_minimum_within_2x(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        non_min = [a for a in result if a.budget > 1.0]
        if len(non_min) >= 2:
            budgets = [a.budget for a in non_min]
            assert max(budgets) <= 2 * min(budgets)


class TestEdgeCaseBudgetEqualsSlots:
    def test_all_get_one(self) -> None:
        league = _league(budget=10)  # exactly 10 slots
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        for alloc in result:
            assert alloc.budget == 1.0

    def test_stars_all_get_one(self) -> None:
        league = _league(budget=10)
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="stars_and_scrubs")
        for alloc in result:
            assert alloc.budget == 1.0


class TestTiersPopulated:
    def test_tiers_set_when_provided(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        tiers = [
            PlayerTier(player_id=1, player_name="Catcher A", position="c", tier=1, value=10.0, rank=1),
            PlayerTier(player_id=2, player_name="Catcher B", position="c", tier=2, value=5.0, rank=2),
            PlayerTier(player_id=7, player_name="Short A", position="ss", tier=1, value=10.0, rank=1),
        ]
        result = optimize_auction_budget(valuations, league, names, tiers=tiers)
        tier_set = [a for a in result if a.target_tier is not None]
        assert len(tier_set) > 0

    def test_no_tiers_when_not_provided(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names)
        for alloc in result:
            assert alloc.target_tier is None


class TestTargetPlayerNames:
    def test_names_populated(self) -> None:
        league = _league()
        valuations, names = _make_valuations()
        result = optimize_auction_budget(valuations, league, names, strategy="balanced")
        for alloc in result:
            assert len(alloc.target_player_names) > 0


def _make_deep_valuations() -> tuple[list[Valuation], dict[int, str]]:
    """Build a large pool suitable for 12-team snake draft simulation.

    Each position gets 15 players with descending values so the pool
    doesn't run out during opponent simulation.
    """
    valuations: list[Valuation] = []
    names: dict[int, str] = {}
    pid = 1

    # Different top values and dropoff rates per position to produce
    # meaningfully different plans for different draft slots.
    # Different top values and dropoff rates per position to produce
    # meaningfully different plans for different draft slots.
    # 30 players per position (180 total) to ensure the pool never
    # runs dry in a 12-team × 10-round (120 pick) simulation.
    batter_specs = {"c": (40.0, 1.2), "1b": (45.0, 1.3), "ss": (50.0, 1.5), "of": (48.0, 1.0)}
    pitcher_specs = {"sp": (47.0, 1.3), "rp": (38.0, 1.1)}

    for pos, (top_val, dropoff) in batter_specs.items():
        for i in range(30):
            val = top_val - i * dropoff
            valuations.append(_valuation(pid, pos, max(val, 1.0)))
            names[pid] = f"{pos.upper()} {i + 1}"
            pid += 1

    for pos, (top_val, dropoff) in pitcher_specs.items():
        for i in range(30):
            val = top_val - i * dropoff
            valuations.append(_valuation(pid, pos, max(val, 1.0), player_type="pitcher"))
            names[pid] = f"{pos.upper()} {i + 1}"
            pid += 1

    return valuations, names


# --- Snake Draft Tests ---


class TestSnakePickNumbers:
    def test_slot_1_in_12_team(self) -> None:
        picks = _snake_pick_numbers(1, 12, 4)
        assert picks == [1, 24, 25, 48]

    def test_slot_12_in_12_team(self) -> None:
        picks = _snake_pick_numbers(12, 12, 4)
        assert picks == [12, 13, 36, 37]

    def test_slot_6_in_12_team(self) -> None:
        picks = _snake_pick_numbers(6, 12, 3)
        assert picks == [6, 19, 30]


class TestPlanCoversAllSlots:
    def test_plan_covers_all_roster_slots(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        total_slots = sum(league.positions.values()) + league.roster_util + sum(league.pitcher_positions.values())
        plan = plan_snake_draft(valuations, league, names, draft_slot=1)
        assert len(plan.rounds) == total_slots

    def test_plan_with_util(self) -> None:
        league = _league(roster_util=1)
        valuations, names = _make_deep_valuations()
        total_slots = sum(league.positions.values()) + league.roster_util + sum(league.pitcher_positions.values())
        plan = plan_snake_draft(valuations, league, names, draft_slot=1)
        assert len(plan.rounds) == total_slots


class TestPickNumbersCorrect:
    def test_slot_1_pick_numbers(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        plan = plan_snake_draft(valuations, league, names, draft_slot=1)
        total_slots = sum(league.positions.values()) + league.roster_util + sum(league.pitcher_positions.values())
        expected_picks = _snake_pick_numbers(1, 12, total_slots)
        actual_picks = [r.pick_number for r in plan.rounds]
        assert actual_picks == expected_picks

    def test_slot_12_pick_numbers(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        plan = plan_snake_draft(valuations, league, names, draft_slot=12)
        total_slots = sum(league.positions.values()) + league.roster_util + sum(league.pitcher_positions.values())
        expected_picks = _snake_pick_numbers(12, 12, total_slots)
        actual_picks = [r.pick_number for r in plan.rounds]
        assert actual_picks == expected_picks


class TestEarlyPicksTargetHighValue:
    def test_round_1_targets_high_value_position(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        plan = plan_snake_draft(valuations, league, names, draft_slot=1)
        first_pos = plan.rounds[0].recommended_position
        # The first pick should be one of the high-value positions
        assert first_pos in {Position.OF, Position.SP, Position.SS, Position.C, Position.FIRST_BASE}


class TestPlanAdaptsToDraftSlot:
    def test_different_slots_different_plans(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        plan_1 = plan_snake_draft(valuations, league, names, draft_slot=1)
        plan_12 = plan_snake_draft(valuations, league, names, draft_slot=12)
        # Different slots produce different pick numbers
        assert plan_1.rounds[0].pick_number != plan_12.rounds[0].pick_number
        # Expected values differ because pool depletion is different
        values_1 = [r.expected_value for r in plan_1.rounds]
        values_12 = [r.expected_value for r in plan_12.rounds]
        assert values_1 != values_12


class TestAllPositionsFilled:
    def test_every_position_targeted(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        plan = plan_snake_draft(valuations, league, names, draft_slot=6)
        targeted_positions = [r.recommended_position for r in plan.rounds]
        # All positions with slots should appear
        for pos, count in league.positions.items():
            assert targeted_positions.count(pos) == count, f"{pos} should appear {count} times"
        for pos, count in league.pitcher_positions.items():
            assert targeted_positions.count(pos) == count, f"{pos} should appear {count} times"


# --- Keeper League Tests ---


class TestKeeperPositionsPreFilled:
    def test_keeper_position_excluded(self) -> None:
        # League with 1 SS slot; keeper at SS means no SS in plan
        league = _league()
        valuations, names = _make_deep_valuations()
        # SS 1 has pid starting at c(30)+1b(30)+1 = 61
        ss_pid = 61  # first SS player
        my_keepers = [(ss_pid, "ss")]
        plan = plan_snake_draft(
            valuations,
            league,
            names,
            draft_slot=1,
            my_keepers=my_keepers,
            keepers_per_team=1,
        )
        positions = [r.recommended_position for r in plan.rounds]
        assert "ss" not in positions


class TestKeptPlayersRemovedFromPool:
    def test_league_keepers_removed(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        # OF starts at pid c(30)+1b(30)+ss(30)+1 = 91; OF 1 is pid 91 with value 48.0
        of_top_pid = 91
        plan = plan_snake_draft(
            valuations,
            league,
            names,
            draft_slot=1,
            league_keeper_ids={of_top_pid},
            keepers_per_team=1,
        )
        # The top OF (value 48.0) should be removed; next OF is value 47.0
        for rnd in plan.rounds:
            if rnd.recommended_position == "of":
                assert rnd.expected_value <= 47.0


class TestFewerRoundsInKeeperMode:
    def test_fewer_rounds(self) -> None:
        league = _league()
        valuations, names = _make_deep_valuations()
        total_slots = sum(league.positions.values()) + league.roster_util + sum(league.pitcher_positions.values())

        plan_redraft = plan_snake_draft(valuations, league, names, draft_slot=1)
        ss_pid = 61
        plan_keeper = plan_snake_draft(
            valuations,
            league,
            names,
            draft_slot=1,
            my_keepers=[(ss_pid, "ss")],
            league_keeper_ids={ss_pid},
            keepers_per_team=1,
        )
        assert len(plan_redraft.rounds) == total_slots
        assert len(plan_keeper.rounds) == total_slots - 1


# --- Monte Carlo Simulation Tests ---


def _sim_league() -> LeagueSettings:
    """4-team league with minimal roster for fast simulations."""
    return LeagueSettings(
        name="Sim Test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=4,
        budget=260,
        roster_batters=3,
        roster_pitchers=1,
        batting_categories=(
            CategoryConfig(key="HR", name="HR", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        pitching_categories=(
            CategoryConfig(key="K", name="K", stat_type=StatType.COUNTING, direction=Direction.HIGHER),
        ),
        positions={Position.C: 1, Position.FIRST_BASE: 1, Position.OF: 1},
        roster_util=1,
    )


def _sim_valuation(player_id: int, position: str, value: float) -> Valuation:
    player_type = "pitcher" if position in ("SP", "RP") else "batter"
    return Valuation(
        player_id=player_id,
        season=2025,
        system="zar",
        version="1.0",
        projection_system="steamer",
        projection_version="1.0",
        player_type=player_type,
        position=position,
        value=value,
        rank=1,
        category_scores={},
    )


def _sim_pool() -> tuple[list[Valuation], dict[int, str]]:
    """Build a pool with enough players for 4 teams × 5 slots = 20 picks."""
    valuations: list[Valuation] = []
    names: dict[int, str] = {}
    positions = ["C", "1B", "OF", "SP"]
    for i in range(40):
        pos = positions[i % len(positions)]
        pid = i + 1
        valuations.append(_sim_valuation(pid, pos, 40.0 - i))
        names[pid] = f"Player {pid}"
    return valuations, names


class TestSimulateDraftsBasic:
    def test_returns_batch_result(self) -> None:
        league = _sim_league()
        valuations, names = _sim_pool()
        result = simulate_drafts(valuations, league, names, draft_slot=1, n_simulations=5, seed=42)
        assert isinstance(result, BatchSimulationResult)
        assert result.summary.n_simulations == 5

    def test_team_idx_matches_slot(self) -> None:
        league = _sim_league()
        valuations, names = _sim_pool()
        result = simulate_drafts(valuations, league, names, draft_slot=3, n_simulations=5, seed=42)
        assert result.summary.team_idx == 2  # 1-indexed slot -> 0-indexed

    def test_deterministic_with_seed(self) -> None:
        league = _sim_league()
        valuations, names = _sim_pool()
        r1 = simulate_drafts(valuations, league, names, draft_slot=1, n_simulations=10, seed=99)
        r2 = simulate_drafts(valuations, league, names, draft_slot=1, n_simulations=10, seed=99)
        assert r1.summary == r2.summary
        assert r1.player_frequencies == r2.player_frequencies

    def test_percentile_ordering(self) -> None:
        league = _sim_league()
        valuations, names = _sim_pool()
        result = simulate_drafts(valuations, league, names, draft_slot=1, n_simulations=20, seed=42)
        s = result.summary
        assert s.p10_roster_value <= s.p25_roster_value
        assert s.p25_roster_value <= s.median_roster_value
        assert s.median_roster_value <= s.p75_roster_value
        assert s.p75_roster_value <= s.p90_roster_value


class TestSimulateDraftsKeepers:
    def test_keeper_excluded_from_pool(self) -> None:
        league = _sim_league()
        valuations, names = _sim_pool()
        # Player 1 is the top player (value 40.0, C)
        result = simulate_drafts(
            valuations,
            league,
            names,
            draft_slot=1,
            n_simulations=10,
            league_keeper_ids={1},
            seed=42,
        )
        # Player 1 should not appear in frequencies with pct > 0
        for f in result.player_frequencies:
            if f.player_id == 1:
                assert f.pct_drafted == 0.0

    def test_keeper_value_added(self) -> None:
        league = _sim_league()
        valuations, names = _sim_pool()
        # Run without keepers
        r_no_keepers = simulate_drafts(valuations, league, names, draft_slot=1, n_simulations=10, seed=42)
        # Run with keeper: Player 1 (C, value 40.0) as my keeper
        r_keepers = simulate_drafts(
            valuations,
            league,
            names,
            draft_slot=1,
            n_simulations=10,
            my_keepers=[(1, "C")],
            league_keeper_ids={1},
            keepers_per_team=1,
            seed=42,
        )
        # Keeper value (40.0) should be added; result should be higher
        assert r_keepers.summary.avg_roster_value > r_no_keepers.summary.avg_roster_value

    def test_fewer_rounds_with_keepers(self) -> None:
        league = _sim_league()
        valuations, names = _sim_pool()
        r_no_keepers = simulate_drafts(valuations, league, names, draft_slot=1, n_simulations=5, seed=42)
        r_keepers = simulate_drafts(
            valuations,
            league,
            names,
            draft_slot=1,
            n_simulations=5,
            my_keepers=[(1, "C")],
            league_keeper_ids={1},
            keepers_per_team=1,
            seed=42,
        )
        # With keepers, fewer players are drafted per simulation
        total_pct_no_k = sum(f.pct_drafted for f in r_no_keepers.player_frequencies)
        total_pct_k = sum(f.pct_drafted for f in r_keepers.player_frequencies)
        assert total_pct_k < total_pct_no_k


# --- _reduce_league_for_keepers Tests ---


class TestReduceLeagueForKeepers:
    def test_pitcher_keeper_reduces_pitcher_slot(self) -> None:
        """Keeper at SP reduces the SP pitcher slot."""
        league = _league()
        result = _reduce_league_for_keepers(league, [(100, "sp")])
        assert result.pitcher_positions[Position.SP] == 1
        assert result.roster_pitchers == league.roster_pitchers - 1

    def test_multiple_pitcher_keepers(self) -> None:
        """Multiple pitcher keepers reduce their respective slots."""
        league = _league()
        result = _reduce_league_for_keepers(league, [(100, "sp"), (101, "rp")])
        assert result.pitcher_positions[Position.SP] == 1
        assert result.pitcher_positions[Position.RP] == 1
        assert result.roster_pitchers == league.roster_pitchers - 2

    def test_batter_keeper_overflows_to_util(self) -> None:
        """Batter keeper with no position slot left uses util."""
        league = _league(
            positions={Position.C: 0, Position.FIRST_BASE: 1, Position.SS: 1, Position.OF: 3}, roster_util=1
        )
        result = _reduce_league_for_keepers(league, [(100, "c")])
        assert result.roster_util == 0
        assert result.roster_batters == league.roster_batters - 1

    def test_pitcher_keeper_falls_back_to_generic_p_slot(self) -> None:
        """Pitcher keeper with exhausted specific slot falls back to generic P."""
        league = _league(pitcher_positions={Position.SP: 0, Position.RP: 0, Position.P: 2})
        result = _reduce_league_for_keepers(league, [(100, "sp")])
        assert result.pitcher_positions[Position.P] == 1
        assert result.roster_pitchers == league.roster_pitchers - 1


# --- simulate_drafts extra_reductions Tests ---


class TestSimulateDraftsExtraReductions:
    def test_keepers_per_team_greater_than_my_keepers(self) -> None:
        """When keepers_per_team > len(my_keepers), extra slots are reduced (lines 493-514)."""
        league = _sim_league()
        valuations, names = _sim_pool()
        # keepers_per_team=3 but only 1 my_keeper → 2 extra reductions
        result = simulate_drafts(
            valuations,
            league,
            names,
            draft_slot=1,
            n_simulations=5,
            my_keepers=[(1, "C")],
            league_keeper_ids={1},
            keepers_per_team=3,
            seed=42,
        )
        assert isinstance(result, BatchSimulationResult)
        # With more keepers_per_team, fewer picks happen overall
        r_1keeper = simulate_drafts(
            valuations,
            league,
            names,
            draft_slot=1,
            n_simulations=5,
            my_keepers=[(1, "C")],
            league_keeper_ids={1},
            keepers_per_team=1,
            seed=42,
        )
        total_pct_3k = sum(f.pct_drafted for f in result.player_frequencies)
        total_pct_1k = sum(f.pct_drafted for f in r_1keeper.player_frequencies)
        assert total_pct_3k < total_pct_1k
