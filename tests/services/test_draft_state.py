from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball_manager.domain.draft_board import DraftBoardRow
from fantasy_baseball_manager.domain.draft_trade import DraftTrade
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

LIVE_CONFIG = DraftConfig(
    teams=4,
    roster_slots={"C": 1, "1B": 1, "OF": 2, "P": 1},
    format=DraftFormat.LIVE,
    user_team=1,
    season=2026,
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
        assert DraftFormat.LIVE == "live"

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

    def test_uppercase_keys(self) -> None:
        """Positions from league config are passed through as-is (already uppercase)."""
        batting_cat = CategoryConfig(
            key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER
        )
        pitching_cat = CategoryConfig(
            key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER
        )
        league = LeagueSettings(
            name="Upper",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=12,
            budget=260,
            roster_batters=10,
            roster_pitchers=5,
            batting_categories=(batting_cat,),
            pitching_categories=(pitching_cat,),
            roster_util=1,
            positions={"SS": 1, "OF": 3, "C": 1},
        )
        slots = build_draft_roster_slots(league)
        assert "SS" in slots
        assert "OF" in slots
        assert "C" in slots
        assert "UTIL" in slots
        assert "P" in slots
        assert all(k == k.upper() for k in slots)

    def test_pitcher_positions_creates_separate_slots(self) -> None:
        """When pitcher_positions is set, create SP/RP/P slots instead of generic P."""
        batting_cat = CategoryConfig(
            key="HR", name="Home Runs", stat_type=StatType.COUNTING, direction=Direction.HIGHER
        )
        pitching_cat = CategoryConfig(
            key="K", name="Strikeouts", stat_type=StatType.COUNTING, direction=Direction.HIGHER
        )
        league = LeagueSettings(
            name="Pitcher Sub-Slots",
            format=LeagueFormat.H2H_CATEGORIES,
            teams=12,
            budget=260,
            roster_batters=10,
            roster_pitchers=8,
            batting_categories=(batting_cat,),
            pitching_categories=(pitching_cat,),
            positions={"C": 1, "1B": 1, "OF": 3},
            roster_util=1,
            pitcher_positions={"SP": 2, "RP": 2, "P": 4},
        )
        slots = build_draft_roster_slots(league)
        assert slots["SP"] == 2
        assert slots["RP"] == 2
        assert slots["P"] == 4
        assert "UTIL" in slots
        # Should NOT have a single P=8 from roster_pitchers
        assert sum(slots[k] for k in ("SP", "RP", "P")) == 8

    def test_empty_pitcher_positions_uses_roster_pitchers(self) -> None:
        """Backward compat: empty pitcher_positions falls back to generic P slot."""
        league = _make_league()
        assert league.pitcher_positions == {}
        slots = build_draft_roster_slots(league)
        assert slots["P"] == 8

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

    def test_pool_keyed_by_player_id_and_type(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        assert (1, "B") in state.available_pool
        assert state.available_pool[(1, "B")].player_name == "Player A"

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

    def test_pick_uses_position_as_given(self) -> None:
        """pick() uses position as-is (callers must pass canonical form)."""
        config = DraftConfig(
            teams=2,
            roster_slots={"C": 1, "OF": 2},
            format=DraftFormat.SNAKE,
            user_team=1,
            season=2026,
        )
        players = [_make_player(1, "P1", "C", 30.0)]
        engine = DraftEngine()
        engine.start(players, config)
        result = engine.pick(player_id=1, team=1, position="C")
        assert result.position == "C"

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
        assert (1, "B") not in state.available_pool
        engine.undo()
        assert (1, "B") in state.available_pool

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
        assert (1, "B") in state.available_pool
        assert (2, "B") in state.available_pool

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


# ---------------------------------------------------------------------------
# Live format — any team in any order
# ---------------------------------------------------------------------------


class TestLiveFormat:
    def test_any_team_any_order(self) -> None:
        """LIVE format allows picks from any team in any order."""
        engine = DraftEngine()
        engine.start(PLAYERS, LIVE_CONFIG)
        # Team 2 picks first, then team 1 — no snake error
        engine.pick(player_id=1, team=2, position="C")
        engine.pick(player_id=2, team=1, position="1B")
        engine.pick(player_id=3, team=4, position="OF")
        engine.pick(player_id=4, team=3, position="OF")
        state = engine.state
        assert len(state.picks) == 4
        assert state.current_pick == 5

    def test_pool_tracking(self) -> None:
        """Pool and roster tracking work in LIVE mode."""
        engine = DraftEngine()
        engine.start(PLAYERS, LIVE_CONFIG)
        engine.pick(player_id=1, team=2, position="C")
        assert (1, "B") not in engine.state.available_pool
        assert len(engine.state.team_rosters[2]) == 1

    def test_team_on_clock_raises(self) -> None:
        """team_on_clock() is not applicable for LIVE format."""
        engine = DraftEngine()
        engine.start(PLAYERS, LIVE_CONFIG)
        with pytest.raises(DraftError, match="not applicable"):
            engine.team_on_clock()

    def test_undo_works(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, LIVE_CONFIG)
        engine.pick(player_id=1, team=3, position="C")
        engine.undo()
        assert (1, "B") in engine.state.available_pool
        assert engine.state.current_pick == 1

    def test_budget_zero_for_live(self) -> None:
        engine = DraftEngine()
        state = engine.start(PLAYERS, LIVE_CONFIG)
        for team in range(1, LIVE_CONFIG.teams + 1):
            assert state.team_budgets[team] == 0


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


# ---------------------------------------------------------------------------
# Pick Trades
# ---------------------------------------------------------------------------


class TestPickTrades:
    def test_team_for_pick_default(self) -> None:
        """Without trades, team_for_pick matches _snake_team."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        for pick in range(1, 9):
            assert engine.team_for_pick(pick) == DraftEngine._snake_team(pick, SNAKE_CONFIG.teams)

    def test_team_for_pick_with_override(self) -> None:
        """Override returns the overridden team."""
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        state.pick_overrides[3] = 1  # Pick 3 now belongs to team 1
        assert engine.team_for_pick(3) == 1

    def test_team_on_clock_respects_override(self) -> None:
        """After trade, team_on_clock returns the new owner."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Pick 1 is team 1 by default. Trade pick 1 (user=team1) for pick 2 (team2)
        engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        # Now pick 1 belongs to team 2
        assert engine.team_on_clock() == 2

    def test_pick_validates_against_override(self) -> None:
        """pick() with original team is rejected; new team is accepted."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Trade: user (team 1) gives pick 1 to team 2, receives pick 2
        engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        # Pick 1 now belongs to team 2 — team 1 is rejected
        with pytest.raises(DraftError, match="expected team 2"):
            engine.pick(player_id=1, team=1, position="C")
        # Team 2 is accepted
        result = engine.pick(player_id=1, team=2, position="C")
        assert result.team == 2

    def test_trade_picks_updates_overrides(self) -> None:
        """Overrides are set correctly after trade."""
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        assert state.pick_overrides[1] == 2
        assert state.pick_overrides[2] == 1

    def test_trade_picks_rejects_already_used_gives(self) -> None:
        """gives picks before current_pick are rejected."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Make pick 1 (team 1's pick)
        engine.pick(player_id=1, team=1, position="C")
        # Pick 1 is already used — can't give it away
        with pytest.raises(DraftError, match="already been used"):
            engine.trade_picks(gives=[1], receives=[2], partner_team=2)

    def test_trade_picks_rejects_already_used_receives(self) -> None:
        """receives picks before current_pick are rejected."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Make picks 1 and 2 (team 1 then team 2)
        engine.pick(player_id=1, team=1, position="C")
        engine.pick(player_id=2, team=2, position="1B")
        # Pick 2 is already used — can't receive it
        # Pick 8 belongs to user (team 1) in 4-team snake round 2
        with pytest.raises(DraftError, match="already been used"):
            engine.trade_picks(gives=[8], receives=[2], partner_team=2)

    def test_trade_picks_rejects_wrong_ownership_gives(self) -> None:
        """gives must belong to user team."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Pick 2 belongs to team 2, not user team 1
        with pytest.raises(DraftError, match="not team"):
            engine.trade_picks(gives=[2], receives=[3], partner_team=3)

    def test_trade_picks_rejects_wrong_ownership_receives(self) -> None:
        """receives must belong to partner team."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Pick 3 belongs to team 3, but we claim partner is team 2
        with pytest.raises(DraftError, match="not partner team"):
            engine.trade_picks(gives=[1], receives=[3], partner_team=2)

    def test_trade_picks_rejects_empty_gives(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        with pytest.raises(DraftError, match="gives must not be empty"):
            engine.trade_picks(gives=[], receives=[2], partner_team=2)

    def test_trade_picks_rejects_empty_receives(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        with pytest.raises(DraftError, match="receives must not be empty"):
            engine.trade_picks(gives=[1], receives=[], partner_team=2)

    def test_undo_trade_restores_ownership(self) -> None:
        """Overrides are removed after undo."""
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        engine.undo_trade()
        assert state.pick_overrides == {}
        assert engine.team_on_clock() == 1  # Back to default

    def test_undo_trade_empty_raises(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        with pytest.raises(DraftError, match="No trades to undo"):
            engine.undo_trade()

    def test_multiple_trades_and_undo(self) -> None:
        """Two trades, undo one, verify partial restore."""
        engine = DraftEngine()
        state = engine.start(PLAYERS, SNAKE_CONFIG)
        # Trade 1: user (team 1) gives pick 1, receives pick 2 from team 2
        engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        # Trade 2: user (now owns pick 2 via override → team 1) gives pick 5,
        # receives pick 3 from team 3
        # Pick 5 in 4-team snake: round 2, position 0 → team 4. User doesn't own pick 5.
        # Pick 8 in 4-team snake: round 2, position 3 → team 1. User owns pick 8.
        engine.trade_picks(gives=[8], receives=[3], partner_team=3)

        assert state.pick_overrides[1] == 2
        assert state.pick_overrides[2] == 1
        assert state.pick_overrides[8] == 3
        assert state.pick_overrides[3] == 1

        # Undo trade 2
        engine.undo_trade()
        assert state.pick_overrides == {1: 2, 2: 1}
        assert 8 not in state.pick_overrides
        assert 3 not in state.pick_overrides

    def test_trades_property(self) -> None:
        """trades property returns list of executed trades."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        assert engine.trades == []
        trade = engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        assert engine.trades == [trade]
        assert isinstance(trade, DraftTrade)

    def test_trade_returns_draft_trade(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        trade = engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        assert trade.team_a == 1
        assert trade.team_b == 2
        assert trade.team_a_gives == [1]
        assert trade.team_b_gives == [2]

    def test_undo_trade_returns_removed_trade(self) -> None:
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        trade = engine.trade_picks(gives=[1], receives=[2], partner_team=2)
        removed = engine.undo_trade()
        assert removed == trade

    def test_trade_between_non_user_teams(self) -> None:
        """team_a parameter allows trades between any two teams."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        # Trade between team 2 and team 3 (user is team 1)
        # Pick 2 belongs to team 2, pick 3 belongs to team 3
        trade = engine.trade_picks(gives=[2], receives=[3], partner_team=3, team_a=2)
        assert trade.team_a == 2
        assert trade.team_b == 3
        assert engine.team_for_pick(2) == 3
        assert engine.team_for_pick(3) == 2


# ---------------------------------------------------------------------------
# Custom Draft Order
# ---------------------------------------------------------------------------


class TestCustomDraftOrder:
    """Tests for draft_order on DraftConfig — custom snake pick ordering."""

    def _make_config(self, draft_order: list[int]) -> DraftConfig:
        return DraftConfig(
            teams=len(draft_order),
            roster_slots={"C": 1, "1B": 1, "OF": 2, "P": 1},
            format=DraftFormat.SNAKE,
            user_team=draft_order[0],
            season=2026,
            draft_order=draft_order,
        )

    def test_round_1_follows_draft_order(self) -> None:
        """With draft_order=[4,3,2,1], pick 1->team 4, pick 2->team 3, etc."""
        config = self._make_config([4, 3, 2, 1])
        engine = DraftEngine()
        engine.start(PLAYERS, config)
        assert engine.team_for_pick(1) == 4
        assert engine.team_for_pick(2) == 3
        assert engine.team_for_pick(3) == 2
        assert engine.team_for_pick(4) == 1

    def test_round_2_snakes_reverse(self) -> None:
        """Round 2 reverses the draft order: [4,3,2,1] -> [1,2,3,4]."""
        config = self._make_config([4, 3, 2, 1])
        engine = DraftEngine()
        engine.start(PLAYERS, config)
        assert engine.team_for_pick(5) == 1
        assert engine.team_for_pick(6) == 2
        assert engine.team_for_pick(7) == 3
        assert engine.team_for_pick(8) == 4

    def test_none_preserves_default(self) -> None:
        """draft_order=None uses standard sequential snake."""
        engine = DraftEngine()
        engine.start(PLAYERS, SNAKE_CONFIG)
        assert SNAKE_CONFIG.draft_order is None
        assert engine.team_for_pick(1) == 1
        assert engine.team_for_pick(4) == 4
        assert engine.team_for_pick(5) == 4  # snake reversal

    def test_picks_validate_against_custom_order(self) -> None:
        """pick() enforces the custom draft order team assignment."""
        config = self._make_config([4, 3, 2, 1])
        engine = DraftEngine()
        engine.start(PLAYERS, config)
        # Pick 1 belongs to team 4 — team 1 is rejected
        with pytest.raises(DraftError, match="expected team 4"):
            engine.pick(player_id=1, team=1, position="C")
        # Team 4 is accepted
        result = engine.pick(player_id=1, team=4, position="C")
        assert result.team == 4

    def test_trades_layer_on_custom_order(self) -> None:
        """Trades override custom draft order assignments."""
        # draft_order=[4,3,2,1], user_team=4 (picks first)
        config = self._make_config([4, 3, 2, 1])
        engine = DraftEngine()
        engine.start(PLAYERS, config)
        # Pick 1 -> team 4 (user), pick 2 -> team 3
        # Trade: user gives pick 1 to team 3, receives pick 2
        engine.trade_picks(gives=[1], receives=[2], partner_team=3)
        assert engine.team_for_pick(1) == 3
        assert engine.team_for_pick(2) == 4
        # Non-traded picks still follow custom order
        assert engine.team_for_pick(3) == 2
        assert engine.team_for_pick(4) == 1

    def test_undo_trade_preserves_custom_order(self) -> None:
        """After undoing a trade, custom draft order base assignments are restored."""
        config = self._make_config([4, 3, 2, 1])
        engine = DraftEngine()
        engine.start(PLAYERS, config)
        engine.trade_picks(gives=[1], receives=[2], partner_team=3)
        engine.undo_trade()
        # Back to custom order
        assert engine.team_for_pick(1) == 4
        assert engine.team_for_pick(2) == 3

    def test_team_on_clock_uses_custom_order(self) -> None:
        """team_on_clock() respects custom draft order."""
        config = self._make_config([4, 3, 2, 1])
        engine = DraftEngine()
        engine.start(PLAYERS, config)
        assert engine.team_on_clock() == 4
        engine.pick(player_id=1, team=4, position="C")
        assert engine.team_on_clock() == 3


# ---------------------------------------------------------------------------
# Two-way player identity tests
# ---------------------------------------------------------------------------


def _make_two_way_pool() -> list[DraftBoardRow]:
    """Create a pool with a two-way player (Ohtani) as both batter and pitcher."""
    return [
        DraftBoardRow(
            player_id=17,
            player_name="Shohei Ohtani",
            rank=1,
            player_type="B",
            position="OF",
            value=50.0,
            category_z_scores={},
        ),
        DraftBoardRow(
            player_id=17,
            player_name="Shohei Ohtani",
            rank=2,
            player_type="P",
            position="SP",
            value=40.0,
            category_z_scores={},
        ),
        DraftBoardRow(
            player_id=1,
            player_name="Player A",
            rank=3,
            player_type="B",
            position="C",
            value=30.0,
            category_z_scores={},
        ),
        DraftBoardRow(
            player_id=2,
            player_name="Player B",
            rank=4,
            player_type="B",
            position="1B",
            value=25.0,
            category_z_scores={},
        ),
    ]


TWO_WAY_CONFIG = DraftConfig(
    teams=2,
    roster_slots={"C": 1, "1B": 1, "OF": 2, "SP": 1, "P": 1},
    format=DraftFormat.LIVE,
    user_team=1,
    season=2026,
)


class TestPickStoresPlayerType:
    def test_pick_stores_player_type_from_pool(self) -> None:
        engine = DraftEngine()
        engine.start(_make_two_way_pool(), TWO_WAY_CONFIG)
        pick = engine.pick(17, team=1, position="OF", player_type="B")
        assert pick.player_type == "B"

    def test_pick_stores_pitcher_type(self) -> None:
        engine = DraftEngine()
        engine.start(_make_two_way_pool(), TWO_WAY_CONFIG)
        pick = engine.pick(17, team=1, position="SP", player_type="P")
        assert pick.player_type == "P"

    def test_pick_without_explicit_type_uses_first_match(self) -> None:
        """When player_type is not specified, the first pool match is used."""
        engine = DraftEngine()
        engine.start(_make_two_way_pool(), TWO_WAY_CONFIG)
        pick = engine.pick(1, team=1, position="C")
        assert pick.player_type == "B"


class TestUndoTwoWayPlayer:
    def test_undo_restores_correct_type(self) -> None:
        """Draft batter-Ohtani then pitcher-Ohtani, undo restores pitcher (last picked)."""
        engine = DraftEngine()
        engine.start(_make_two_way_pool(), TWO_WAY_CONFIG)

        # Team 1 drafts batter-Ohtani
        engine.pick(17, team=1, position="OF", player_type="B")
        # Team 2 drafts pitcher-Ohtani
        engine.pick(17, team=2, position="SP", player_type="P")

        # Both Ohtani entries should be gone from pool
        assert (17, "B") not in engine.state.available_pool
        assert (17, "P") not in engine.state.available_pool

        # Undo the last pick (pitcher-Ohtani)
        undone = engine.undo()
        assert undone.player_type == "P"

        # Pitcher-Ohtani should be back in the pool, batter still gone
        assert (17, "P") in engine.state.available_pool
        assert (17, "B") not in engine.state.available_pool

    def test_undo_legacy_pick_without_player_type(self) -> None:
        """A pick with empty player_type falls back to player_id search."""
        engine = DraftEngine()
        engine.start(_make_two_way_pool(), TWO_WAY_CONFIG)

        # Pick normally so it goes into _removed_rows
        engine.pick(1, team=1, position="C")

        # Simulate a legacy pick by replacing the last pick with one missing player_type
        last = engine.state.picks[-1]
        legacy = DraftPick(
            pick_number=last.pick_number,
            team=last.team,
            player_id=last.player_id,
            player_name=last.player_name,
            position=last.position,
            player_type="",  # empty — legacy
            price=None,
        )
        engine.state.picks[-1] = legacy
        engine.state.team_rosters[1][-1] = legacy

        # Undo should still work via player_id fallback
        undone = engine.undo()
        assert undone.player_id == 1
        assert (1, "B") in engine.state.available_pool
