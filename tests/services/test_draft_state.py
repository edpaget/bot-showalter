from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.league_settings import (
    CategoryConfig,
    Direction,
    LeagueFormat,
    LeagueSettings,
    StatType,
)
from fantasy_baseball_manager.services.draft_state import (
    DraftConfig,
    DraftEngine,
    DraftError,
    DraftFormat,
    DraftPick,
    DraftState,
    build_draft_roster_slots,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_player(player_id: int, name: str, position: str, value: float) -> DraftBoardRow:
    return DraftBoardRow(
        player_id=player_id,
        player_name=name,
        rank=player_id,
        player_type="B" if position != "SP" else "P",
        position=position,
        value=value,
        category_z_scores={},
    )


PLAYERS = [
    _make_player(1, "Player A", "C", 30.0),
    _make_player(2, "Player B", "1B", 25.0),
    _make_player(3, "Player C", "OF", 20.0),
    _make_player(4, "Player D", "OF", 18.0),
    _make_player(5, "Player E", "SP", 15.0),
    _make_player(6, "Player F", "C", 12.0),
    _make_player(7, "Player G", "OF", 10.0),
    _make_player(8, "Player H", "1B", 8.0),
]

SNAKE_CONFIG = DraftConfig(
    teams=4,
    roster_slots={"C": 1, "1B": 1, "OF": 2, "P": 1},
    format=DraftFormat.SNAKE,
    user_team=1,
    season=2026,
)

AUCTION_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"C": 1, "1B": 1, "OF": 1},
    format=DraftFormat.AUCTION,
    user_team=1,
    season=2026,
    budget=10,
)


def _make_league() -> LeagueSettings:
    batting_cat = CategoryConfig(key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
    pitching_cat = CategoryConfig(key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER)
    return LeagueSettings(
        name="Test League",
        format=LeagueFormat.H2H_CATEGORIES,
        teams=12,
        budget=260,
        roster_batters=10,
        roster_pitchers=8,
        batting_categories=(batting_cat,),
        pitching_categories=(pitching_cat,),
        roster_util=1,
        positions={"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "MI": 1, "CI": 1},
    )


# ---------------------------------------------------------------------------
# Step 1: Types — frozen/mutable, defaults, enum values
# ---------------------------------------------------------------------------


class TestTypes:
    def test_draft_format_values(self) -> None:
        assert DraftFormat.SNAKE == "snake"
        assert DraftFormat.AUCTION == "auction"

    def test_draft_error_is_exception(self) -> None:
        with pytest.raises(DraftError):
            raise DraftError("boom")

    def test_draft_config_is_frozen(self) -> None:
        config = DraftConfig(
            teams=12,
            roster_slots={"C": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        with pytest.raises(FrozenInstanceError):
            config.teams = 10  # type: ignore[misc]

    def test_draft_config_budget_default(self) -> None:
        config = DraftConfig(
            teams=12,
            roster_slots={"C": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        assert config.budget == 0

    def test_draft_pick_is_frozen(self) -> None:
        pick = DraftPick(
            pick_number=1,
            team=1,
            player_id=100,
            player_name="Test",
            position="C",
        )
        with pytest.raises(FrozenInstanceError):
            pick.team = 2  # type: ignore[misc]

    def test_draft_pick_price_default_none(self) -> None:
        pick = DraftPick(
            pick_number=1,
            team=1,
            player_id=100,
            player_name="Test",
            position="C",
        )
        assert pick.price is None

    def test_draft_state_is_mutable(self) -> None:
        state = DraftState(
            config=SNAKE_CONFIG,
            picks=[],
            available_pool={},
            team_rosters={},
            team_budgets={},
        )
        state.current_pick = 5
        assert state.current_pick == 5


# ---------------------------------------------------------------------------
# Step 2: build_draft_roster_slots
# ---------------------------------------------------------------------------


class TestBuildDraftRosterSlots:
    def test_standard_league(self) -> None:
        league = _make_league()
        slots = build_draft_roster_slots(league)
        assert slots["C"] == 1
        assert slots["1B"] == 1
        assert slots["OF"] == 3
        assert slots["UTIL"] == 1
        assert slots["P"] == 8

    def test_no_util(self) -> None:
        league = _make_league()
        # Override roster_util to 0 by creating new instance
        league_no_util = LeagueSettings(
            name=league.name,
            format=league.format,
            teams=league.teams,
            budget=league.budget,
            roster_batters=league.roster_batters,
            roster_pitchers=league.roster_pitchers,
            batting_categories=league.batting_categories,
            pitching_categories=league.pitching_categories,
            roster_util=0,
            positions=league.positions,
        )
        slots = build_draft_roster_slots(league_no_util)
        assert "UTIL" not in slots

    def test_empty_positions(self) -> None:
        league = LeagueSettings(
            name="Empty",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=12,
            budget=260,
            roster_batters=0,
            roster_pitchers=5,
            batting_categories=(),
            pitching_categories=(),
            roster_util=0,
            positions={},
        )
        slots = build_draft_roster_slots(league)
        assert slots == {"P": 5}


# ---------------------------------------------------------------------------
# Step 3: start()
# ---------------------------------------------------------------------------


class TestStart:
    def test_initializes_pool(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        assert len(engine.available()) == len(PLAYERS)

    def test_pool_keyed_by_player_id(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        assert 1 in state.available_pool
        assert state.available_pool[1].player_name == "Player A"

    def test_empty_rosters(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        for team in range(1, SNAKE_CONFIG.teams + 1):
            assert state.team_rosters[team] == []

    def test_budget_init_auction(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, AUCTION_CONFIG)
        for team in range(1, AUCTION_CONFIG.teams + 1):
            assert state.team_budgets[team] == AUCTION_CONFIG.budget

    def test_budget_zero_for_snake(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        for team in range(1, SNAKE_CONFIG.teams + 1):
            assert state.team_budgets[team] == 0

    def test_current_pick_starts_at_1(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        assert state.current_pick == 1

    def test_pick_before_start_raises(self) -> None:
        engine = DraftEngine()
        with pytest.raises(DraftError, match="not started"):
            engine.pick(player_id=1, team=1, position="C")


# ---------------------------------------------------------------------------
# Step 4: pick() — basic snake
# ---------------------------------------------------------------------------


class TestPickBasicSnake:
    def test_returns_draft_pick(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        result = engine.pick(player_id=1, team=1, position="C")
        assert isinstance(result, DraftPick)
        assert result.pick_number == 1
        assert result.team == 1
        assert result.player_id == 1
        assert result.player_name == "Player A"
        assert result.position == "C"
        assert result.price is None

    def test_pick_removes_from_pool(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        avail = engine.available()
        assert all(p.player_id != 1 for p in avail)

    def test_pick_adds_to_roster(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        assert len(state.team_rosters[1]) == 1
        assert state.team_rosters[1][0].player_id == 1

    def test_pick_advances_counter(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        assert state.current_pick == 2

    def test_pick_added_to_picks_list(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        assert len(state.picks) == 1


# ---------------------------------------------------------------------------
# Step 5: pick() — pool validation
# ---------------------------------------------------------------------------


class TestPickPoolValidation:
    def test_reject_unknown_player(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        with pytest.raises(DraftError, match="not in the available pool"):
            engine.pick(player_id=999, team=1, position="C")

    def test_reject_already_picked_player(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        with pytest.raises(DraftError, match="not in the available pool"):
            engine.pick(player_id=1, team=2, position="C")


# ---------------------------------------------------------------------------
# Step 6: pick() — roster constraints
# ---------------------------------------------------------------------------


class TestPickRosterConstraints:
    def test_reject_full_position_simple(self) -> None:
        """Config has C: 1. Second C pick for same team should fail."""
        config = DraftConfig(
            teams=2,
            roster_slots={"C": 1, "OF": 2},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        players = [
            _make_player(1, "C1", "C", 30.0),
            _make_player(2, "C2", "C", 20.0),
            _make_player(3, "OF1", "OF", 25.0),
            _make_player(4, "OF2", "OF", 15.0),
        ]
        engine = DraftEngine()
        engine.start(players, config)
        engine.pick(player_id=1, team=1, position="C")
        engine.pick(player_id=3, team=2, position="OF")
        # Round 2 snake: team 2, then team 1
        engine.pick(player_id=4, team=2, position="OF")
        # Now back to team 1 — C slot is full, player 2 is still available
        with pytest.raises(DraftError, match="roster slot.*full"):
            engine.pick(player_id=2, team=1, position="C")

    def test_reject_full_position_slot(self) -> None:
        """Directly test slot-full check on team 1."""
        config = DraftConfig(
            teams=2,
            roster_slots={"C": 1, "OF": 5},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        players = [
            _make_player(1, "C1", "C", 30.0),
            _make_player(2, "OF1", "OF", 25.0),
            _make_player(3, "OF2", "OF", 20.0),
            _make_player(4, "C2", "C", 15.0),
            _make_player(5, "OF3", "OF", 10.0),
            _make_player(6, "OF4", "OF", 5.0),
        ]
        engine = DraftEngine()
        engine.start(players, config)
        # Pick 1: team 1 drafts C
        engine.pick(player_id=1, team=1, position="C")
        # Pick 2: team 2 drafts OF
        engine.pick(player_id=2, team=2, position="OF")
        # Pick 3 (round 2 snake): team 2
        engine.pick(player_id=3, team=2, position="OF")
        # Pick 4: team 1 — try to draft another C when slot is full
        with pytest.raises(DraftError, match="roster slot.*full"):
            engine.pick(player_id=4, team=1, position="C")

    def test_reject_unknown_position_slot(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        with pytest.raises(DraftError, match="not a valid roster slot"):
            engine.pick(player_id=1, team=1, position="DH")


# ---------------------------------------------------------------------------
# Step 7: Snake ordering — team_on_clock + validation
# ---------------------------------------------------------------------------


class TestSnakeOrdering:
    def test_team_on_clock_round_1(self) -> None:
        """4-team snake: round 1 is teams 1,2,3,4."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        assert engine.team_on_clock() == 1
        engine.pick(player_id=1, team=1, position="C")
        assert engine.team_on_clock() == 2
        engine.pick(player_id=2, team=2, position="1B")
        assert engine.team_on_clock() == 3
        engine.pick(player_id=3, team=3, position="OF")
        assert engine.team_on_clock() == 4

    def test_team_on_clock_round_2(self) -> None:
        """4-team snake: round 2 reverses to 4,3,2,1."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Complete round 1
        engine.pick(player_id=1, team=1, position="C")
        engine.pick(player_id=2, team=2, position="1B")
        engine.pick(player_id=3, team=3, position="OF")
        engine.pick(player_id=4, team=4, position="OF")
        # Round 2
        assert engine.team_on_clock() == 4
        engine.pick(player_id=5, team=4, position="P")
        assert engine.team_on_clock() == 3
        engine.pick(player_id=7, team=3, position="1B")
        assert engine.team_on_clock() == 2
        engine.pick(player_id=8, team=2, position="OF")
        assert engine.team_on_clock() == 1

    def test_wrong_team_rejected(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        with pytest.raises(DraftError, match="expected team 1"):
            engine.pick(player_id=1, team=3, position="C")

    def test_3_team_snake(self) -> None:
        """Odd-team snake: 3 teams, rounds 1→2 should be 1,2,3,3,2,1."""
        config = DraftConfig(
            teams=3,
            roster_slots={"C": 1, "OF": 5},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        players = [_make_player(i, f"P{i}", "OF", 30.0 - i) for i in range(1, 7)]
        engine = DraftEngine()
        engine.start(players, config)
        assert engine.team_on_clock() == 1
        engine.pick(player_id=1, team=1, position="OF")
        assert engine.team_on_clock() == 2
        engine.pick(player_id=2, team=2, position="OF")
        assert engine.team_on_clock() == 3
        engine.pick(player_id=3, team=3, position="OF")
        # Round 2: 3,2,1
        assert engine.team_on_clock() == 3
        engine.pick(player_id=4, team=3, position="OF")
        assert engine.team_on_clock() == 2
        engine.pick(player_id=5, team=2, position="OF")
        assert engine.team_on_clock() == 1

    def test_team_on_clock_not_started_raises(self) -> None:
        engine = DraftEngine()
        with pytest.raises(DraftError, match="not started"):
            engine.team_on_clock()


# ---------------------------------------------------------------------------
# Step 8: Auction budget
# ---------------------------------------------------------------------------


class TestAuctionBudget:
    def test_budget_deducted(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, AUCTION_CONFIG)
        engine.pick(player_id=1, team=1, position="C", price=5)
        assert state.team_budgets[1] == 5

    def test_missing_price_rejected(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        with pytest.raises(DraftError, match="price.*required"):
            engine.pick(player_id=1, team=1, position="C")

    def test_overspend_rejected(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        with pytest.raises(DraftError, match="exceeds.*budget"):
            engine.pick(player_id=1, team=1, position="C", price=11)

    def test_reserve_requirement(self) -> None:
        """Must reserve $1 per remaining slot after this pick."""
        # Budget=10, 3 slots. After 1 pick, need $1×2 remaining = $2 reserved.
        # So max spend on first pick is 10 - 2 = 8.
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        with pytest.raises(DraftError, match="reserve"):
            engine.pick(player_id=1, team=1, position="C", price=9)

    def test_reserve_allows_max_valid(self) -> None:
        """Budget=10, 3 slots. First pick can spend up to $8."""
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        pick = engine.pick(player_id=1, team=1, position="C", price=8)
        assert pick.price == 8

    def test_last_slot_can_spend_remaining(self) -> None:
        """When filling the last slot, no reserve needed."""
        engine = DraftEngine()
        state = engine.start(PLAYERS, AUCTION_CONFIG)
        engine.pick(player_id=1, team=1, position="C", price=4)
        engine.pick(player_id=2, team=1, position="1B", price=1)
        # Last slot: remaining budget is 10-4-1 = 5, and 0 slots remain after
        engine.pick(player_id=3, team=1, position="OF", price=5)
        assert state.team_budgets[1] == 0

    def test_auction_any_team_order(self) -> None:
        """In auction, any team can pick in any order."""
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        engine.pick(player_id=1, team=2, position="C", price=5)
        engine.pick(player_id=2, team=1, position="1B", price=3)
        # No ordering error — both succeeded

    def test_price_recorded_on_pick(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        pick = engine.pick(player_id=1, team=1, position="C", price=5)
        assert pick.price == 5

    def test_zero_price_rejected(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)
        with pytest.raises(DraftError, match="at least \\$1"):
            engine.pick(player_id=1, team=1, position="C", price=0)


# ---------------------------------------------------------------------------
# Step 9: undo()
# ---------------------------------------------------------------------------


class TestUndo:
    def test_undo_restores_pool(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        assert 1 not in state.available_pool
        engine.undo()
        assert 1 in state.available_pool

    def test_undo_restores_roster(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        engine.undo()
        assert state.team_rosters[1] == []

    def test_undo_restores_pick_counter(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        engine.undo()
        assert state.current_pick == 1

    def test_undo_removes_from_picks(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        engine.undo()
        assert state.picks == []

    def test_undo_returns_undone_pick(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        undone = engine.undo()
        assert undone.player_id == 1

    def test_undo_empty_raises(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        with pytest.raises(DraftError, match="no picks to undo"):
            engine.undo()

    def test_multi_undo(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        engine.pick(player_id=2, team=2, position="1B")
        engine.undo()
        engine.undo()
        assert state.current_pick == 1
        assert len(state.picks) == 0
        assert 1 in state.available_pool
        assert 2 in state.available_pool

    def test_undo_restores_auction_budget(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, AUCTION_CONFIG)
        engine.pick(player_id=1, team=1, position="C", price=5)
        assert state.team_budgets[1] == 5
        engine.undo()
        assert state.team_budgets[1] == 10

    def test_undo_not_started_raises(self) -> None:
        engine = DraftEngine()
        with pytest.raises(DraftError, match="not started"):
            engine.undo()


# ---------------------------------------------------------------------------
# Step 10: available()
# ---------------------------------------------------------------------------


class TestAvailable:
    def test_returns_all_sorted_by_value(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        avail = engine.available()
        values = [p.value for p in avail]
        assert values == sorted(values, reverse=True)

    def test_filter_by_position(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        catchers = engine.available(position="C")
        assert len(catchers) == 2
        assert all(p.position == "C" for p in catchers)

    def test_filter_empty_result(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        result = engine.available(position="SS")
        assert result == []


# ---------------------------------------------------------------------------
# Step 11: my_roster()
# ---------------------------------------------------------------------------


class TestMyRoster:
    def test_returns_user_picks_only(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        engine.pick(player_id=2, team=2, position="1B")
        roster = engine.my_roster()
        assert len(roster) == 1
        assert roster[0].player_id == 1

    def test_empty_roster(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        assert engine.my_roster() == []

    def test_not_started_raises(self) -> None:
        engine = DraftEngine()
        with pytest.raises(DraftError, match="not started"):
            engine.my_roster()


# ---------------------------------------------------------------------------
# Step 12: my_needs()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 12a: state property
# ---------------------------------------------------------------------------


class TestStateProperty:
    def test_returns_state_after_start(self) -> None:
        engine = DraftEngine()
        started = engine.start(PLAYERS, SNAKE_CONFIG)
        assert engine.state is started

    def test_raises_before_start(self) -> None:
        engine = DraftEngine()
        with pytest.raises(DraftError, match="not started"):
            engine.state  # noqa: B018


# ---------------------------------------------------------------------------
# Step 12b: my_needs()
# ---------------------------------------------------------------------------


class TestMyNeeds:
    def test_all_slots_unfilled(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        needs = engine.my_needs()
        assert needs == {"C": 1, "1B": 1, "OF": 2, "P": 1}

    def test_partially_filled(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        engine.pick(player_id=1, team=1, position="C")
        needs = engine.my_needs()
        assert "C" not in needs
        assert needs["1B"] == 1

    def test_not_started_raises(self) -> None:
        engine = DraftEngine()
        with pytest.raises(DraftError, match="not started"):
            engine.my_needs()


# ---------------------------------------------------------------------------
# Step 13: Integration tests
# ---------------------------------------------------------------------------


class TestIntegrationSnake:
    def test_full_mini_draft(self) -> None:
        """4 teams, 2 rounds (8 picks total) snake draft."""
        config = DraftConfig(
            teams=4,
            roster_slots={"C": 1, "OF": 1},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        players = [_make_player(i, f"P{i}", "C" if i <= 4 else "OF", 30.0 - i) for i in range(1, 9)]
        engine = DraftEngine()
        engine.start(players, config)

        # Round 1: teams 1,2,3,4
        engine.pick(player_id=1, team=1, position="C")
        engine.pick(player_id=2, team=2, position="C")
        engine.pick(player_id=3, team=3, position="C")
        engine.pick(player_id=4, team=4, position="C")

        # Round 2 (snake reversal): teams 4,3,2,1
        engine.pick(player_id=5, team=4, position="OF")
        engine.pick(player_id=6, team=3, position="OF")
        engine.pick(player_id=7, team=2, position="OF")
        engine.pick(player_id=8, team=1, position="OF")

        state = engine._state
        assert state is not None
        assert state.current_pick == 9
        assert len(state.picks) == 8
        assert len(state.available_pool) == 0
        for team in range(1, 5):
            assert len(state.team_rosters[team]) == 2

        # User team (1) has picks 1 and 8
        roster = engine.my_roster()
        assert [p.player_id for p in roster] == [1, 8]
        assert engine.my_needs() == {}


class TestIntegrationAuction:
    def test_full_mini_auction(self) -> None:
        """2 teams, 3 slots each, budget 10."""
        engine = DraftEngine()
        engine.start(PLAYERS, AUCTION_CONFIG)

        # Team 1 picks
        engine.pick(player_id=1, team=1, position="C", price=4)
        engine.pick(player_id=2, team=2, position="1B", price=3)
        engine.pick(player_id=3, team=1, position="OF", price=4)
        engine.pick(player_id=4, team=2, position="OF", price=5)
        engine.pick(player_id=5, team=2, position="C", price=2)
        engine.pick(player_id=8, team=1, position="1B", price=2)

        state = engine._state
        assert state is not None
        assert len(state.picks) == 6
        assert state.team_budgets[1] == 0
        assert state.team_budgets[2] == 0

        roster = engine.my_roster()
        assert len(roster) == 3
        assert engine.my_needs() == {}
