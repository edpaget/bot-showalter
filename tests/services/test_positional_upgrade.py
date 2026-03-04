import pytest

from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.domain.positional_upgrade import (
    MarginalValue,
    RosterSlot,
    RosterState,
)
from fantasy_baseball_manager.services.positional_upgrade import (
    build_roster_state,
    compute_marginal_values,
    compute_opportunity_costs,
    compute_position_upgrades,
)

# -- Helpers ------------------------------------------------------------------


def _cat(key: str) -> CategoryConfig:
    return CategoryConfig(key=key, name=key, stat_type=StatType.COUNTING, direction=Direction.HIGHER)


BATTING_CATS = (_cat("HR"), _cat("SB"), _cat("AVG"))
PITCHING_CATS = (_cat("W"), _cat("K"))


def _league(
    positions: dict[str, int] | None = None,
    roster_util: int = 1,
    roster_pitchers: int = 0,
    pitcher_positions: dict[str, int] | None = None,
) -> LeagueSettings:
    return LeagueSettings(
        name="test",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=10,
        roster_pitchers=roster_pitchers,
        batting_categories=BATTING_CATS,
        pitching_categories=PITCHING_CATS,
        positions=positions or {"C": 1, "SS": 1, "OF": 3},
        roster_util=roster_util,
        pitcher_positions=pitcher_positions or {},
    )


def _row(
    player_id: int,
    name: str,
    position: str,
    value: float,
    z_scores: dict[str, float] | None = None,
    player_type: str = "batter",
) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type=player_type,
        position=position,
        value=value,
        category_z_scores=z_scores or {},
    )


def _board(rows: list[DraftBoardRow]) -> DraftBoard:
    return DraftBoard(
        rows=rows,
        batting_categories=("HR", "SB", "AVG"),
        pitching_categories=("W", "K"),
    )


# -- build_roster_state tests ------------------------------------------------


class TestBuildRosterStateEmpty:
    """Empty roster → all slots open."""

    def test_all_slots_open(self) -> None:
        league = _league()
        board = _board([])
        state = build_roster_state([], board, league)

        # C=1, SS=1, OF=3, UTIL=1 → 6 slots
        assert len(state.slots) == 6
        assert state.total_value == 0.0
        assert all(s.player_id is None for s in state.slots)

    def test_open_positions_listed(self) -> None:
        league = _league()
        board = _board([])
        state = build_roster_state([], board, league)
        assert sorted(state.open_positions) == ["C", "OF", "OF", "OF", "SS", "UTIL"]


class TestBuildRosterStateSinglePlayer:
    """One drafted player fills the matching slot."""

    def test_slot_filled(self) -> None:
        row = _row(1, "Adley Rutschman", "C", 15.0, {"HR": 0.5, "SB": -0.2, "AVG": 0.8})
        league = _league()
        board = _board([row])

        state = build_roster_state([1], board, league)

        filled = [s for s in state.slots if s.player_id is not None]
        assert len(filled) == 1
        assert filled[0].position == "C"
        assert filled[0].player_name == "Adley Rutschman"
        assert filled[0].value == 15.0

    def test_category_totals(self) -> None:
        row = _row(1, "Adley Rutschman", "C", 15.0, {"HR": 0.5, "SB": -0.2, "AVG": 0.8})
        league = _league()
        board = _board([row])

        state = build_roster_state([1], board, league)
        assert state.category_totals == {"HR": 0.5, "SB": -0.2, "AVG": 0.8}

    def test_total_value(self) -> None:
        row = _row(1, "Adley Rutschman", "C", 15.0, {"HR": 0.5})
        league = _league()
        board = _board([row])

        state = build_roster_state([1], board, league)
        assert state.total_value == 15.0


class TestBuildRosterStateMultiPosition:
    """Multi-position player assigned to scarcer position."""

    def test_assigned_to_scarcer_slot(self) -> None:
        # SS has 1 slot, OF has 3 → SS is scarcer
        row = _row(10, "Multi Guy", "OF", 20.0)
        league = _league(positions={"SS": 1, "OF": 3})
        board = _board([row])
        eligibility = {10: ["SS", "OF"]}

        state = build_roster_state([10], board, league, position_eligibility=eligibility)
        filled = [s for s in state.slots if s.player_id is not None]
        assert len(filled) == 1
        assert filled[0].position == "SS"


class TestBuildRosterStateUtilFallback:
    """Player fills UTIL when specific position is full."""

    def test_util_fallback(self) -> None:
        row_c1 = _row(1, "Catcher One", "C", 15.0)
        row_c2 = _row(2, "Catcher Two", "C", 10.0)
        league = _league(positions={"C": 1, "SS": 1, "OF": 3}, roster_util=1)
        board = _board([row_c1, row_c2])

        state = build_roster_state([1, 2], board, league)
        filled = [s for s in state.slots if s.player_id is not None]
        assert len(filled) == 2
        positions_filled = {s.position for s in filled}
        assert "C" in positions_filled
        assert "UTIL" in positions_filled


class TestBuildRosterStatePitcherSlots:
    """Pitcher fills P slot from roster_pitchers."""

    def test_pitcher_slot(self) -> None:
        row = _row(50, "Ace Pitcher", "SP", 25.0, {"W": 1.0, "K": 1.5}, player_type="pitcher")
        league = _league(roster_pitchers=5)
        board = _board([row])

        state = build_roster_state([50], board, league)
        filled = [s for s in state.slots if s.player_id is not None]
        assert len(filled) == 1
        assert filled[0].position == "P"
        assert filled[0].player_name == "Ace Pitcher"


class TestBuildRosterStatePitcherPositions:
    """Pitcher fills specific pitcher position slot from pitcher_positions."""

    def test_specific_pitcher_position(self) -> None:
        row = _row(50, "Ace Pitcher", "SP", 25.0, {"W": 1.0}, player_type="pitcher")
        league = _league(
            pitcher_positions={"SP": 2, "RP": 2},
            roster_pitchers=0,
        )
        board = _board([row])

        state = build_roster_state([50], board, league)
        filled = [s for s in state.slots if s.player_id is not None]
        assert len(filled) == 1
        assert filled[0].position == "SP"


class TestBuildRosterStateCategoryTotals:
    """Category totals sum correctly across multiple players."""

    def test_sums_across_players(self) -> None:
        rows = [
            _row(1, "A", "C", 10.0, {"HR": 1.0, "SB": 0.5}),
            _row(2, "B", "SS", 12.0, {"HR": 0.3, "SB": -0.2}),
        ]
        league = _league()
        board = _board(rows)

        state = build_roster_state([1, 2], board, league)
        assert state.category_totals["HR"] == pytest.approx(1.3)
        assert state.category_totals["SB"] == pytest.approx(0.3)


# -- compute_marginal_values tests -------------------------------------------


class TestComputeMarginalValuesOpenSlot:
    """Player at open position scores higher than same-value player at filled position."""

    def test_open_position_boosted(self) -> None:
        # Roster: C filled, SS open, no UTIL — so C Guy can only upgrade
        c_slot = RosterSlot(
            position="C",
            player_id=1,
            player_name="Catcher",
            value=15.0,
            category_z_scores={"HR": 0.5, "SB": 0.5, "AVG": 0.5},
        )
        ss_slot = RosterSlot(position="SS")
        state = RosterState(
            slots=[c_slot, ss_slot],
            open_positions=["SS"],
            total_value=15.0,
            category_totals={"HR": 0.0, "SB": 0.0, "AVG": 0.0},
        )

        # Two players with same value — one at open SS, one at filled C
        available = [
            _row(10, "SS Guy", "SS", 20.0, {"HR": 0.5, "SB": 0.5, "AVG": 0.5}),
            _row(11, "C Guy", "C", 20.0, {"HR": 0.5, "SB": 0.5, "AVG": 0.5}),
        ]
        league = _league(positions={"C": 1, "SS": 1}, roster_util=0)

        results = compute_marginal_values(state, available, league)
        ss_result = next(r for r in results if r.player_id == 10)
        c_result = next(r for r in results if r.player_id == 11)

        assert ss_result.fills_need is True
        assert c_result.fills_need is False
        assert ss_result.marginal_value > c_result.marginal_value


class TestComputeMarginalValuesCategoryNeed:
    """Player improving weak category scores higher than one piling onto strong."""

    def test_weak_category_bonus(self) -> None:
        slot = RosterSlot(
            position="C",
            player_id=1,
            player_name="Catcher",
            value=15.0,
            category_z_scores={"HR": 2.0, "SB": -1.5, "AVG": 0.0},
        )
        ss_slot = RosterSlot(position="SS")
        state = RosterState(
            slots=[slot, ss_slot],
            open_positions=["SS"],
            total_value=15.0,
            category_totals={"HR": 2.0, "SB": -1.5, "AVG": 0.0},
        )

        # Player A: strong in SB (weak cat), weak in HR (strong cat)
        # Player B: strong in HR (strong cat), weak in SB (weak cat)
        available = [
            _row(10, "Speed Guy", "SS", 18.0, {"HR": 0.0, "SB": 2.0, "AVG": 0.0}),
            _row(11, "Power Guy", "SS", 18.0, {"HR": 2.0, "SB": 0.0, "AVG": 0.0}),
        ]
        league = _league(positions={"C": 1, "SS": 1})

        results = compute_marginal_values(state, available, league)
        speed = next(r for r in results if r.player_id == 10)
        power = next(r for r in results if r.player_id == 11)

        # Speed Guy should rank higher because SB is a weak category
        assert speed.marginal_value > power.marginal_value


class TestComputeMarginalValuesMultiPosition:
    """Multi-position player fills whichever gives higher marginal value."""

    def test_optimal_assignment(self) -> None:
        # SS filled, 2B open
        ss_slot = RosterSlot(position="SS", player_id=1, player_name="Starter", value=10.0)
        two_b_slot = RosterSlot(position="2B")
        state = RosterState(
            slots=[ss_slot, two_b_slot],
            open_positions=["2B"],
            total_value=10.0,
            category_totals={"HR": 0.0, "SB": 0.0},
        )

        available = [
            _row(20, "Combo Player", "SS", 15.0, {"HR": 0.5, "SB": 0.5}),
        ]
        league = _league(positions={"SS": 1, "2B": 1}, roster_util=0)
        eligibility = {20: ["SS", "2B"]}

        results = compute_marginal_values(state, available, league, position_eligibility=eligibility)
        assert len(results) == 1
        result = results[0]
        # Should fill the open 2B slot, not try to upgrade SS
        assert result.fills_need is True
        assert result.position == "2B"


class TestComputeMarginalValuesNoSlot:
    """Player with no available slot gets marginal_value = 0."""

    def test_no_fit(self) -> None:
        # All slots filled, no UTIL
        c_slot = RosterSlot(position="C", player_id=1, player_name="Catcher", value=20.0)
        state = RosterState(
            slots=[c_slot],
            open_positions=[],
            total_value=20.0,
            category_totals={"HR": 0.0},
        )
        # Another C available but slot is full and no upgrade possible
        available = [_row(10, "Worse Catcher", "C", 10.0, {"HR": 0.1})]
        league = _league(positions={"C": 1}, roster_util=0)

        results = compute_marginal_values(state, available, league)
        assert len(results) == 1
        assert results[0].marginal_value == 0.0


class TestComputeMarginalValuesUpgrade:
    """Player that upgrades an existing starter records the upgrade."""

    def test_upgrade_recorded(self) -> None:
        c_slot = RosterSlot(
            position="C",
            player_id=1,
            player_name="Weak Catcher",
            value=5.0,
            category_z_scores={"HR": 0.1},
        )
        state = RosterState(
            slots=[c_slot],
            open_positions=[],
            total_value=5.0,
            category_totals={"HR": 0.1},
        )

        available = [_row(10, "Better Catcher", "C", 20.0, {"HR": 1.5})]
        league = _league(positions={"C": 1}, roster_util=0)

        results = compute_marginal_values(state, available, league)
        assert len(results) == 1
        result = results[0]
        assert result.fills_need is False
        assert result.upgrade_over == "Weak Catcher"
        assert result.marginal_value > 0


class TestComputeMarginalValuesSorted:
    """Results are sorted by marginal_value descending."""

    def test_descending_order(self) -> None:
        ss_slot = RosterSlot(position="SS")
        of_slot = RosterSlot(position="OF")
        state = RosterState(
            slots=[ss_slot, of_slot],
            open_positions=["SS", "OF"],
            total_value=0.0,
            category_totals={"HR": 0.0},
        )

        available = [
            _row(1, "Low", "SS", 5.0, {"HR": 0.1}),
            _row(2, "High", "OF", 30.0, {"HR": 2.0}),
            _row(3, "Mid", "SS", 15.0, {"HR": 0.5}),
        ]
        league = _league(positions={"SS": 1, "OF": 1}, roster_util=0)

        results = compute_marginal_values(state, available, league)
        values = [r.marginal_value for r in results]
        assert values == sorted(values, reverse=True)


# -- compute_position_upgrades tests ----------------------------------------


class TestPositionUpgradesOpenHighDropoff:
    """Open position with large dropoff → 'high' urgency."""

    def test_high_urgency(self) -> None:
        state = RosterState(
            slots=[RosterSlot(position="C"), RosterSlot(position="SS")],
            open_positions=["C", "SS"],
            total_value=0.0,
            category_totals={},
        )
        available = [
            _row(1, "Great C", "C", 20.0),
            _row(2, "OK C", "C", 10.0),
        ]
        league = _league(positions={"C": 1, "SS": 1}, roster_util=0)

        results = compute_position_upgrades(state, available, league)
        c_upgrade = next(r for r in results if r.position == "C")
        assert c_upgrade.urgency == "high"
        assert c_upgrade.current_player is None
        assert c_upgrade.best_available == "Great C"
        assert c_upgrade.best_available_value == 20.0
        assert c_upgrade.upgrade_value == 20.0
        assert c_upgrade.dropoff_to_next == 10.0


class TestPositionUpgradesOpenSmallDropoff:
    """Open position with small dropoff → 'medium' urgency."""

    def test_medium_urgency(self) -> None:
        state = RosterState(
            slots=[RosterSlot(position="C")],
            open_positions=["C"],
            total_value=0.0,
            category_totals={},
        )
        available = [
            _row(1, "C1", "C", 20.0),
            _row(2, "C2", "C", 18.0),
        ]
        league = _league(positions={"C": 1}, roster_util=0)

        results = compute_position_upgrades(state, available, league)
        assert len(results) == 1
        assert results[0].urgency == "medium"
        assert results[0].dropoff_to_next == pytest.approx(2.0)


class TestPositionUpgradesFilledWithUpgrade:
    """Filled position, upgrade available → 'low' urgency, positive upgrade_value."""

    def test_low_urgency_positive_upgrade(self) -> None:
        state = RosterState(
            slots=[
                RosterSlot(position="C", player_id=1, player_name="Starter C", value=10.0),
            ],
            open_positions=[],
            total_value=10.0,
            category_totals={},
        )
        available = [
            _row(5, "Better C", "C", 18.0),
            _row(6, "OK C", "C", 12.0),
        ]
        league = _league(positions={"C": 1}, roster_util=0)

        results = compute_position_upgrades(state, available, league)
        assert len(results) == 1
        assert results[0].urgency == "low"
        assert results[0].upgrade_value == pytest.approx(8.0)
        assert results[0].current_player == "Starter C"


class TestPositionUpgradesFilledCurrentBetter:
    """Filled position, current > best → 'low' urgency, negative upgrade_value."""

    def test_negative_upgrade(self) -> None:
        state = RosterState(
            slots=[
                RosterSlot(position="C", player_id=1, player_name="Elite C", value=30.0),
            ],
            open_positions=[],
            total_value=30.0,
            category_totals={},
        )
        available = [_row(5, "Mediocre C", "C", 10.0)]
        league = _league(positions={"C": 1}, roster_util=0)

        results = compute_position_upgrades(state, available, league)
        assert len(results) == 1
        assert results[0].upgrade_value == pytest.approx(-20.0)
        assert results[0].urgency == "low"


class TestPositionUpgradesDropoff:
    """Dropoff computed correctly between #1 and #2."""

    def test_dropoff_between_top_two(self) -> None:
        state = RosterState(
            slots=[RosterSlot(position="SS")],
            open_positions=["SS"],
            total_value=0.0,
            category_totals={},
        )
        available = [
            _row(1, "SS1", "SS", 25.0),
            _row(2, "SS2", "SS", 17.0),
        ]
        league = _league(positions={"SS": 1}, roster_util=0)

        results = compute_position_upgrades(state, available, league)
        assert results[0].dropoff_to_next == pytest.approx(8.0)
        assert results[0].next_best == "SS2"


class TestPositionUpgradesSingleAvailable:
    """Single available player → next_best=None, dropoff=best_value."""

    def test_single_player(self) -> None:
        state = RosterState(
            slots=[RosterSlot(position="C")],
            open_positions=["C"],
            total_value=0.0,
            category_totals={},
        )
        available = [_row(1, "Only C", "C", 15.0)]
        league = _league(positions={"C": 1}, roster_util=0)

        results = compute_position_upgrades(state, available, league)
        assert results[0].next_best is None
        assert results[0].dropoff_to_next == pytest.approx(15.0)


class TestPositionUpgradesMultiEligibility:
    """Multi-position player appears in both position checks."""

    def test_multi_position_player(self) -> None:
        state = RosterState(
            slots=[RosterSlot(position="SS"), RosterSlot(position="2B")],
            open_positions=["SS", "2B"],
            total_value=0.0,
            category_totals={},
        )
        available = [_row(1, "Combo", "SS", 20.0)]
        league = _league(positions={"SS": 1, "2B": 1}, roster_util=0)
        eligibility = {1: ["SS", "2B"]}

        results = compute_position_upgrades(state, available, league, position_eligibility=eligibility)
        # Player should appear as best_available for both positions
        positions = {r.position for r in results}
        assert "SS" in positions
        assert "2B" in positions
        ss = next(r for r in results if r.position == "SS")
        two_b = next(r for r in results if r.position == "2B")
        assert ss.best_available == "Combo"
        assert two_b.best_available == "Combo"


class TestPositionUpgradesSortOrder:
    """Output sorted: high → medium → low, then by upgrade_value desc."""

    def test_sort_order(self) -> None:
        state = RosterState(
            slots=[
                RosterSlot(position="C"),
                RosterSlot(position="SS"),
                RosterSlot(position="OF", player_id=99, player_name="OF Starter", value=15.0),
            ],
            open_positions=["C", "SS"],
            total_value=15.0,
            category_totals={},
        )
        available = [
            # C: open, big dropoff → high
            _row(1, "Great C", "C", 25.0),
            _row(2, "OK C", "C", 10.0),
            # SS: open, small dropoff → medium
            _row(3, "SS1", "SS", 20.0),
            _row(4, "SS2", "SS", 18.0),
            # OF: filled → low
            _row(5, "Better OF", "OF", 22.0),
        ]
        league = _league(positions={"C": 1, "SS": 1, "OF": 1}, roster_util=0)

        results = compute_position_upgrades(state, available, league)

        urgency_order = {"high": 0, "medium": 1, "low": 2}
        for i in range(len(results) - 1):
            a, b = results[i], results[i + 1]
            if urgency_order[a.urgency] == urgency_order[b.urgency]:
                assert a.upgrade_value >= b.upgrade_value
            else:
                assert urgency_order[a.urgency] <= urgency_order[b.urgency]


# -- compute_opportunity_costs tests ----------------------------------------


def _mv(
    player_id: int,
    name: str,
    position: str,
    marginal_value: float,
    fills_need: bool,
) -> MarginalValue:
    return MarginalValue(
        player_id=player_id,
        player_name=name,
        position=position,
        raw_value=marginal_value,
        marginal_value=marginal_value,
        category_impacts={},
        fills_need=fills_need,
    )


def _empty_state() -> RosterState:
    return RosterState(slots=[], open_positions=[], total_value=0.0, category_totals={})


class TestOpportunityCostNoNonFillCompetition:
    """Position fill with no non-fill players in window → opp cost 0, draft now."""

    def test_no_competition(self) -> None:
        mvs = [
            _mv(1, "Catcher A", "C", 20.0, fills_need=True),
            _mv(2, "Catcher B", "C", 15.0, fills_need=True),
        ]
        league = _league(positions={"C": 1}, roster_util=0)
        results = compute_opportunity_costs(mvs, _empty_state(), league, picks_until_next=2)
        assert len(results) == 2
        assert results[0].opportunity_cost == 0.0
        assert results[0].net_value == 20.0
        assert results[0].recommendation == "draft now"


class TestOpportunityCostEliteNonFillInWindow:
    """Elite non-fill in window → opportunity_cost > MV → 'wait'."""

    def test_wait_when_elite_alternative(self) -> None:
        mvs = [
            _mv(1, "Elite OF", "OF", 30.0, fills_need=False),
            _mv(2, "OK Catcher", "C", 15.0, fills_need=True),
            _mv(3, "Weak SS", "SS", 5.0, fills_need=True),
        ]
        league = _league(positions={"C": 1, "SS": 1, "OF": 3}, roster_util=0)
        results = compute_opportunity_costs(mvs, _empty_state(), league, picks_until_next=2)
        catcher = next(r for r in results if r.position == "C")
        assert catcher.opportunity_cost == 30.0
        assert catcher.net_value < 0
        assert catcher.recommendation == "wait"


class TestOpportunityCostWeakerNonFillInWindow:
    """Weaker non-fill in window → net_value > 0 → 'draft now'."""

    def test_draft_now_when_alternative_weaker(self) -> None:
        mvs = [
            _mv(1, "Great Catcher", "C", 25.0, fills_need=True),
            _mv(2, "OK OF", "OF", 10.0, fills_need=False),
            _mv(3, "Weak 1B", "1B", 5.0, fills_need=True),
        ]
        league = _league(positions={"C": 1, "1B": 1, "OF": 3}, roster_util=0)
        results = compute_opportunity_costs(mvs, _empty_state(), league, picks_until_next=2)
        catcher = next(r for r in results if r.position == "C")
        assert catcher.opportunity_cost == 10.0
        assert catcher.net_value == pytest.approx(15.0)
        assert catcher.recommendation == "draft now"


class TestOpportunityCostExactTie:
    """net_value == 0 → 'borderline'."""

    def test_borderline(self) -> None:
        mvs = [
            _mv(1, "OF Star", "OF", 20.0, fills_need=False),
            _mv(2, "Catcher", "C", 20.0, fills_need=True),
        ]
        league = _league(positions={"C": 1, "OF": 3}, roster_util=0)
        results = compute_opportunity_costs(mvs, _empty_state(), league, picks_until_next=2)
        catcher = next(r for r in results if r.position == "C")
        assert catcher.net_value == 0.0
        assert catcher.recommendation == "borderline"


class TestOpportunityCostWindowSize:
    """picks_until_next controls the window — larger window changes result."""

    def test_window_controls_result(self) -> None:
        mvs = [
            _mv(1, "Fill C", "C", 18.0, fills_need=True),
            _mv(2, "Non-fill A", "OF", 12.0, fills_need=False),
            _mv(3, "Non-fill B", "OF", 25.0, fills_need=False),
        ]
        league = _league(positions={"C": 1, "OF": 3}, roster_util=0)

        # Window=1: only top-1 player (Fill C itself) in gone set,
        # no non-fill in gone set → opp cost = 0
        small = compute_opportunity_costs(mvs, _empty_state(), league, picks_until_next=1)
        c_small = next(r for r in small if r.position == "C")
        assert c_small.opportunity_cost == 0.0

        # Window=3: all 3 in gone set, best non-fill = Non-fill B (25.0)
        big = compute_opportunity_costs(mvs, _empty_state(), league, picks_until_next=3)
        c_big = next(r for r in big if r.position == "C")
        assert c_big.opportunity_cost == 25.0


class TestOpportunityCostSortedByNetValue:
    """Output is sorted by net_value descending."""

    def test_sort_order(self) -> None:
        mvs = [
            _mv(1, "Non-fill", "OF", 15.0, fills_need=False),
            _mv(2, "Fill Low", "C", 10.0, fills_need=True),
            _mv(3, "Fill High", "SS", 25.0, fills_need=True),
        ]
        league = _league(positions={"C": 1, "SS": 1, "OF": 3}, roster_util=0)
        results = compute_opportunity_costs(mvs, _empty_state(), league, picks_until_next=3)
        net_values = [r.net_value for r in results]
        assert net_values == sorted(net_values, reverse=True)
